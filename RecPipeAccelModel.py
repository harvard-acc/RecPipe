#!/usr/bin/env python3
import json
import math
from collections import OrderedDict
import sys
import numpy as np
import time

from collections import deque

class LRUCache:
    def __init__(self, capacity):
        self.cache = {}
        self.capacity = capacity
        self.access_list = deque()

    def get(self, key, track_lru=False):
        if key not in self.cache:
            return None
        else:
            if track_lru:
                self.access_list.remove(key)
                self.access_list.append(key)

            return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.access_list.append(key)
        if len(self.cache) > self.capacity:
            remove_key = self.access_list.popleft()
            del self.cache[remove_key]

############################################################################
# Memory system for RecPipeAccelerator
############################################################################
class Memory:
    def __init__(self, accelerator, config):
        self.accelerator = accelerator

        # DRAM parameters
        self.dram_latency   = config["dram_latency"]
        self.dram_bandwidth = config["dram_bandwidth"]
        self.dram_capacity  = config["dram_capacity"]

        # Flash parameters
        self.flash_latency   = config["flash_latency"]
        self.flash_bandwidth = config["flash_bandwidth"]
        self.flash_capacity  = config["flash_capacity"]

        emb_vector_size = self.accelerator.elementsize * self.accelerator.featuresize
        self.l1_lru     = LRUCache(config["l1_cache_capacity"] / emb_vector_size)
        self.dram_lru   = LRUCache(self.dram_capacity * 2.0**30 / emb_vector_size)

    # Warm up embedding cache
    def warm_cache(self, indices):
        for index in indices.flatten():
            if self.l1_lru.get(index):
                continue
            else:
                self.l1_lru.put(index, True)

            if self.dram_lru.get(index):
                continue
            else:
                self.dram_lru.put(index, True)

    # Access embeddings
    def access_embeddings(self, items_per_query, indices,
                          characterization="offline",
                          print_debug=False):

        # Multi-level hierarchy with input-dependant caching behavior
        if characterization=="online":
            l1_hits = 0
            l1_misses = 0
            dram_hits = 0
            dram_misses = 0
            for index in indices.flatten():
                if self.l1_lru.get(index):
                    l1_hits += 1
                else:
                    l1_misses += 1
                    self.l1_lru.put(index, True)
                    if self.dram_lru.get(index):
                        dram_hits += 1
                    else:
                        dram_misses += 1
                        self.dram_lru.put(index, True)
            l1_missrate = float(l1_misses) / \
                (self.accelerator.lookups_per_item * self.accelerator.indices_per_lookup * items_per_query)

            dram_missrate = float(dram_misses) / \
                (self.accelerator.lookups_per_item * self.accelerator.indices_per_lookup * items_per_query)

        else:

            # L1 miss rates based on off-line embedding cache characterization
            l1_missrate   = 0.50
            dram_missrate = 0.00

        # Simple DRAM bandwidth model
        if self.dataset == "kaggle":
            bytes = self.accelerator.elementsize * self.accelerator.featuresize * self.accelerator.indices_per_lookup * \
                self.accelerator.lookups_per_item * items_per_query
        elif "movielens" in self.dataset:
            bytes = self.accelerator.elementsize * self.accelerator.featuresize * self.accelerator.indices_per_lookup * \
                (self.accelerator.lookups_per_item / 2) * items_per_query

            bytes += self.accelerator.elementsize * self.accelerator.mf_featuresize * self.accelerator.indices_per_lookup * \
                (self.accelerator.lookups_per_item / 2) * items_per_query

        dram_bytes = bytes * l1_missrate
        dram_ns = dram_bytes / (float(self.dram_bandwidth) * 2**30 / 10**9)
        if bytes > 0:
            dram_cycles = self.dram_latency + (dram_ns / self.accelerator.clock)
        else:
            dram_cycles = (dram_ns / self.accelerator.clock)

        sram_cycles = bytes * (1-l1_missrate) * 1 / 64

        # Simple FLASH bandwidth model
        if dram_missrate > 0:
            flash_bytes  = bytes * dram_missrate
            flash_ns     = flash_bytes / (float(self.flash_bandwidth) * 2**30 / 10**9)
            flash_cycles = self.flash_latency + flash_ns / self.accelerator.clock
        else:
            flash_cycles = 0

        cycles = max(flash_cycles, dram_cycles) + sram_cycles

        if print_debug:
            print("Flash vs. dram vs. sram", flash_cycles, dram_cycles, sram_cycles)

        return {"latency" : cycles, "energy" : 0}

    def load_weights(self, bytes):
        # Simple DRAM bandwidth model
        ns = bytes / (float(self.dram_bandwidth) * 2**30 / 10**9)
        cycles = self.dram_latency + (ns / self.accelerator.clock)

        return {"latency" : cycles, "energy" : 0}


############################################################################
# Processing embeddings based on on-chip embedding cache
############################################################################
class EmbeddingUnit:
    def __init__(self, accelerator, memory, config):
        self.accelerator = accelerator
        self.memory = memory

    # Access embeddings from embedding-cache
    def access_embeddings(self, items_per_query, indices):
        return self.memory.access_embeddings(items_per_query, indices)

    # Helper function to warm-up embedding cache
    def warm_cache(self, indices):
        self.memory.warm_cache(indices)


############################################################################
# Processing element of MLP layers based on systolic arrays.
############################################################################
class ProcessingElement:
    def __init__(self, memory, config):
        self.memory = memory

        # Array size for systolic array (units in terms of number of MACs in
        # single column/row)
        self.arraysize = config["arraysize"]

        # Systolic array weight SRAM sizes
        self.weight_sram_size = config["weight_sram_size"]
        self.weight_sram_latency = config["weight_sram_latency"]

        # Systolic array output activation SRAM sizes
        self.output_sram_size = config["output_sram_size"]
        self.output_sram_latency = config["output_sram_latency"]

    def process_item(self, layers, items_per_query, print_debug=False):

        def layer_latency(layer_in, layer_out, items_per_query):
            if print_debug:
                print("Processing layer size of: ", layer_in, layer_out)

            ##################################################################
            # Weight loading latency
            ##################################################################
            # Total size of weight parameters assuming 4 bytes per element
            weight_bytes = 4 * layer_in * layer_out

            # Assuming no double buffering
            num_weight_slices = math.ceil(float(weight_bytes) / (self.weight_sram_size))

            # Amount of data to load is minimum of SRAM size and weights size
            loading_bytes = min(self.weight_sram_size, weight_bytes)

            loading_latency = self.memory.load_weights(loading_bytes)["latency"]
            loading_latency = loading_latency * num_weight_slices

            ##################################################################
            # Computation latency
            ##################################################################
            # The number of tiles we need is based on how much we need to tile
            # the input and output dimensions
            sa_tiles = math.ceil(layer_in / float(self.arraysize)) * \
                math.ceil(layer_out / float(self.arraysize))
            sa_tiles = int(sa_tiles)

            # Tile latency is time it takes to width (min of layer_out, array_size)
            #                                  + height (min of layer_in, array_size)
            #                                  * batch_size (number of items)
            #                                  * number of tiles
            sa_latency = (min(layer_out, self.arraysize) + min(layer_in, self.arraysize)) + items_per_query
            sa_latency = sa_latency * sa_tiles

            in_dim_tiles = math.ceil(layer_in / float(self.arraysize))

            reduction_cycles = int(layer_out + in_dim_tiles) / 8

            if print_debug:
                print("Loading latency of weights is: ", loading_latency)
                print("Reduction_cycle: ", reduction_cycles)
                print("Computation latency is is: ", sa_latency)

                print("Loading latency", loading_latency)
                print("SA latency", sa_latency)
                print("reduction_cycles", reduction_cycles)

            latency = loading_latency + sa_latency

            return latency

        # Layer by layer we calculate the time it takes to compute MLP
        # operations.
        cycles = 0
        flops = 0
        for i in range(len(layers)-1):
            flops += layers[i] * layers[i+1]
            cycles += layer_latency(layers[i], layers[i+1], items_per_query)

        if print_debug:
            print("Flops: ", flops)

        return {"latency" : cycles, "energy" : 0}


# On-chip sorting unit for RecPipeAccel
class SortingUnit:
    def __init__(self, accelerator, memory, config, on_chip_sorting):
        self.accelerator = accelerator
        self.memory = memory

        self.on_chip_sorting = on_chip_sorting

    def sort_items(self, items_per_query, top_n):
        if self.on_chip_sorting:
            # For RecPipeAccel sorting is perfomed on-chip. Performance of
            # sorting unit is based on offline, synthesized RTL
            # characteriation.
            sram_word_width = 64
            cycles = items_per_query / (sram_word_width / self.accelerator.elementsize)
        else:
            # For baseline accelerators sorting is performed on the host CPU
            if items_per_query < 512:
                cycles = 0.03 * 1000 * 1000 # 0.03ms
            elif items_per_query <= 1024:
                cycles = 0.07 * 1000 * 1000 # 0.07ms
            elif items_per_query <= 4096:
                cycles = 0.35 * 1000 * 1000 # 0.35ms

        return {"latency" : cycles, "energy" : 0}


"""
Analytical, cycle-approximate model of multi-stage recommendation inference
accelerator, RecPipeAccel.

Accelerator considers tradeoffs between monolithic versus splitting
accelerators into multiple sub-units.

Current RecPipeAccel can be extended to consider different hardware topologies.

RecPipeAccel is validated with other cycle-approximate accelerator models
(e.g., ScaleSim) and synthesized RTL.
"""
class Accelerator:
    def __init__(self, model_configfile, hardware_configfile, dataset="kaggle",):

        # Model configuration file
        with open(model_configfile, 'r') as configfile:
            self.model_config = json.load(configfile)

        # Hardware accelerator configuration file
        with open(hardware_configfile, 'r') as configfile:
            self.config = json.load(configfile)

        # Sparse vector embedding feature size
        self.featuresize = self.model_config["arch_sparse_feature_size"]

        # dataset (kaggle | movielens1m | movielens20m )
        self.dataset = dataset

        if "movielens" in self.dataset:
            self.mf_featuresize = self.model_config["arch_sparse_mf_feature_size"]

        # MLP layers in model
        mlp_bot = self.model_config["arch_mlp_bot"].split("-")
        mlp_top = self.model_config["arch_mlp_top"].split("-")

        self.mlp_bot = list(map(lambda x: int(x), mlp_bot))
        self.mlp_top = list(map(lambda x: int(x), mlp_top))

        # Dataset-specific embedding dimensions
        if dataset == "kaggle":
            self.elementsize = 4
            self.indices_per_lookup = 1
            self.lookups_per_item = 26
            self.dataset = "kaggle"
        elif dataset == "movielens20m":
            self.elementsize = 4
            self.indices_per_lookup = 1
            self.lookups_per_item = 4
            self.dataset = "movielens20m"
        elif dataset == "movielens1m":
            self.elementsize = 4
            self.indices_per_lookup = 1
            self.lookups_per_item = 4
            self.dataset = "movielens1m"

        # Clock frequency (units in ns)
        self.clock = self.config["clock"]

        # Memory unit (DRAM)
        self.memory = Memory(self, self.config)

        # On-chip embedding cache for hot embeddings
        self.embedding_unit = EmbeddingUnit(self, self.memory, self.config)

        # Compute units/PEs following systolic arrays
        self.pe_list = [ProcessingElement(self.memory, self.config) for i in range(self.config["pe_count"])]

        self.memory.dataset = self.dataset
        self.embedding_unit.dataset = self.dataset

        # RecPipeAccel performs sorting (based on user-item rankings) on-chip
        self.on_chip_sorting = True

        self.sorting_unit = SortingUnit(self, self.memory, self.config, self.on_chip_sorting)

    # Helper function for embedding cache warm up
    def warm_cache(self, embedding_indices):
        self.embedding_unit.warm_cache(embedding_indices)

    # Helper function quantifying the PCIe communication overheads for sending
    # inputs between CPU host to RecPipeAccel
    def input_communication_latency(self, items_per_query,):
        # PCIe latency based on offline measurements/characterization of Intel
        # Cascade Lake to NVIDIA T4 GPU. Units in ms.
        pcie_latency = 5

        # PCIe bandwidth based on offline measurements/characterization of Intel
        # Cascade Lake to NVIDIA T4 GPU.
        # bw_slope is in ms per items_per_query
        bw_slope = 12./4096.
        pcie_bw =  bw_slope * items_per_query # pcie_bw in ms

        # total PCIe communication
        data_communication_latency = pcie_latency + pcie_bw
        return data_communication_latency

    # Helper function quantifying the PCIe communication overheads for sending
    # inputs between RecPipeAccel to CPU host
    def output_communication_latency(self, items_per_query,):
        # PCIe latency based on offline measurements/characterization of Intel
        # Cascade Lake to NVIDIA T4 GPU. Units in ms.
        pcie_latency = 5

        # PCIe bandwidth based on offline measurements/characterization of Intel
        # Cascade Lake to NVIDIA T4 GPU.
        # bw_slope is in ms per items_per_query
        bw_slope = 5./4096.
        pcie_bw =  bw_slope * items_per_query

        # total PCIe communication
        data_communication_latency = pcie_latency + pcie_bw
        return data_communication_latency


    # Run query for accelerator inference
    def run_query(self, items_per_query, top_n, embedding_indices, print_debug=False):
        # PCIe input latency
        input_pcie_latency = self.input_communication_latency(items_per_query) * 1000.

        # Embedding latency
        embedding_latency = self.embedding_unit.access_embeddings(items_per_query, embedding_indices)["latency"]

        # Compute MLP (compute operation) latency
        mlp_bot = self.mlp_bot
        if self.dataset == "kaggle":
            mlp_top = [self.featuresize * (self.lookups_per_item+1)] + self.mlp_top
        elif "movielens" in self.dataset:
            mlp_top = self.mlp_top

        # Systolic array latencies validated based on offline characterization
        # on ScaleSim and synthesized RTL
        pe_latency_bot = self.pe_list[0].process_item(mlp_bot, items_per_query)["latency"]
        pe_latency_top = self.pe_list[0].process_item(mlp_top, items_per_query)["latency"]

        # Sort latency
        sort_latency = self.sorting_unit.sort_items(items_per_query, top_n)["latency"]
        sort_latency = sort_latency / self.clock

        if self.on_chip_sorting:
            # Output PCIe latency based on sorted user-items
            output_pcie_latency = self.output_communication_latency(top_n)*1000.
        else:
            # Need to send all predicted CTR's back to host CPU
            output_pcie_latency = self.output_communication_latency(items_per_query)*1000.

        # Aggregate all component latencies
        latency =  0
        latency += input_pcie_latency
        latency += embedding_latency
        latency += pe_latency_bot
        latency += pe_latency_top
        latency += sort_latency
        latency += output_pcie_latency

        if print_debug:
            print("***Input PCIe: "     , input_pcie_latency)
            print("***MLP Bot Latency: ", pe_latency_bot)
            print("***MLP Top Latency: ", pe_latency_top)
            print("***Embedding latency", embedding_latency)
            print("***Sorting latency: ", sort_latency)
            print("***Output PCIe: "    , output_pcie_latency)

        latency_s = latency * (self.clock) / 10**9

        scores          = 0
        preprocess_time = 0
        data_times      = 0
        inference_times = latency_s
        fetch_times     = 0

        return (scores, preprocess_time, data_times, inference_times, fetch_times)

    # Helper function for model configuration
    def print_model_config(self, ):
        print("Model configuration")

        print("***MLP Bottom: ", self.mlp_bot)
        print("***MLP Top: ", self.mlp_top)
        print("***Datatype width (bytes): ", self.elementsize)
        print("***Embedding vector dimension: ", self.featuresize)
        print("***Indices per lookup: ", self.indices_per_lookup)
        print("***lookups_per_item: ", self.lookups_per_item)
        return

