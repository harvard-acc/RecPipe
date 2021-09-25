import os, sys
import time
import json

import numpy as np
from numpy.random import choice

from collections import deque
from tqdm import tqdm

# Generate Lookup IDs for 2-Stage Recommendation Pipeline
def generate_ids(file, filter_ratio, randomize=False):
    ids_all     = np.load(file).reshape(-1, 26)
    num_samples = ids_all.shape[0]

    ids_1 = ids_all

    if randomize:
        ids_2 = ids_all[choice(num_samples, int(num_samples*filter_ratio), replace=False)]
    else:
        ids_2 = ids_all[0:int(num_samples*filter_ratio)]

    return [ids_1, ids_2]

# Generate On-Chip Cache Sizes for 2-Stage Recommendation Pipeline
def generate_cache_entries(cap_total, cap_ratio, emb_bytes, cache_line_size):
    # Original: each cache line fits one embedding vector
    # return (cap_total * np.array([cap_ratio, 1-cap_ratio]) / emb_bytes).astype(int) # Returns number of entries for each cache
    
    # Updated: each cache line has cache_line_size bytes
    return (cap_total * np.array([cap_ratio, 1-cap_ratio]) / cache_line_size).astype(int) # Returns number of entries for each cache

# On-Chip Cache
class LRUCache:
    def __init__(self, capacity):
        self.cache       = {}
        self.capacity    = capacity
        self.access_list = deque()

        self.accesses = 0
        self.hits     = 0
        self.warmup   = False # Warmup mode to prepopulate the cache

    def put(self, key, value):
        self.cache[key] = value
        self.access_list.append(key)

        if len(self.cache) > self.capacity:
            remove_key = self.access_list.popleft()
            del self.cache[remove_key]

    def get(self, key):
        # Warmup Mode: On
        if self.warmup:
            if key in self.cache:
                pass
            else:
                self.put(key, None)
        # Warmup Mode: Off
        else:
            self.accesses += 1
            if key in self.cache:
                self.hits += 1
            else:
                self.put(key, None)

def main():
    # Make Results Directory
    RESULTS_DIR = 'RPAccel_characterization/embedding/results'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # External Data Trace
    indices_file = './RPAccel_characterization/embedding/indices_lookup_small.npy' # or wherever you store your indices
    
    # Model Architecture Parameters
    precision       = 32 # FP32
    epochs          = 1 # Epochs for training set
    emb_dims        = np.array([4, 32]) # Dimension of each embedding vector
    emb_bytes       = emb_dims * (precision/8) # Bytes per embedding vector
    large_table_ids = [2, 3, 11, 15, 20] # Large embedding tables
    table_copies    = 4
    filter_ratios   = [0.25, 0.2, 0.125, 0.1, 0.0625] # 2-Stage pipeline filtering ratios

    # Cache Parameters
    cache_capacities_total = np.array([1, 2, 4, 6, 8, 12, 16]) * (1024**2) # Total cache capacities in bytes
    cache_ratios           = np.array([0.0, 0.01, 0.03, 0.05, 0.07, 
                                       0.09, 0.1, 0.2, 0.3, 0.4, 
                                       0.5, 0.6, 0.7, 0.8, 0.9, 
                                       0.91, 0.93, 0.95, 0.97, 0.99, 1.0]) # Fraction of total cache dedicated for first stage
    cache_line_size        = 128 # Cache line size (Bytes)

    total_experiments = len(filter_ratios) * len(cache_capacities_total) * len(cache_ratios)
    experiment_num    = 1

    for filter_ratio in filter_ratios:
        for cache_capacity_total in cache_capacities_total:
            for cache_ratio in cache_ratios:

                print('*** {}/{}: Filter Ratio: {}, Total Cache Size: {}MB, Cache Ratio: {}'.format(experiment_num, total_experiments, filter_ratio, cache_capacity_total/(1024**2), cache_ratio))

                # Generate IDs
                indices_1, indices_2 = generate_ids(indices_file, filter_ratio, True)

                # Generate Caches
                entries_1, entries_2 = generate_cache_entries(cache_capacity_total, cache_ratio, emb_bytes, cache_line_size)
                lru_1 = LRUCache(entries_1)
                lru_2 = LRUCache(entries_2)

                # Warmup Cache 1
                lru_1.warmup = True
                for i in tqdm(range(indices_1.shape[0]), desc='Cache 1 Warmup', leave=True):
                    for j in large_table_ids:
                        for k in range(table_copies):
                            lru_1.get((j, k, indices_1[i][j]))
                # Warmup Cache 2
                lru_2.warmup = True
                for i in tqdm(range(indices_2.shape[0]), desc='Cache 2 Warmup', leave=True):
                    for j in large_table_ids:
                        for k in range(table_copies):
                            lru_2.get((j, k, indices_1[i][j]))

                # Evaluate Cache 1
                lru_1.warmup = False
                for _ in range(epochs):
                    for i in tqdm(range(indices_1.shape[0]), desc='Cache 1 Evaluation', leave=True):
                        for j in large_table_ids:
                            for k in range(table_copies):
                                lru_1.get((j, k, indices_1[i][j]))
                # Evaluate Cache 2
                lru_2.warmup = False
                for _ in range(epochs):
                    for i in tqdm(range(indices_2.shape[0]), desc='Cache 2 Evaluation', leave=True):
                        for j in large_table_ids:
                            for k in range(table_copies):
                                lru_2.get((j, k, indices_1[i][j]))

                total_accesses   = lru_1.accesses + lru_2.accesses
                total_hits       = lru_1.hits + lru_2.hits
                total_bytes      = lru_1.accesses * emb_bytes[0] + lru_2.accesses * emb_bytes[1]
                total_bytes_sram = lru_1.hits * emb_bytes[0] + lru_2.hits * emb_bytes[1] # Only counts useful bytes.

                
                # Write results
                outfile = './{}/{}_{}_{}.json'.format(RESULTS_DIR, filter_ratio, int(cache_capacity_total/(1024**2)), cache_ratio)

                results = {}

                results['filter_ratio']         = float(filter_ratio)
                results['accesses_total']       = int(total_accesses)
                results['accesses_1']           = int(lru_1.accesses)
                results['accesses_2']           = int(lru_2.accesses)

                results['cache_capacity_total'] = int(cache_capacity_total)
                results['cache_capacity_1']     = int(entries_1 * emb_bytes[0])
                results['cache_capacity_2']     = int(entries_2 * emb_bytes[1])
                results['cache_ratio']          = float(cache_ratio)
                results['cache_entries_total']  = int(entries_1 + entries_2)
                results['cache_entries_1']      = int(entries_1)
                results['cache_entries_2']      = int(entries_2)

                results['hits_1']               = int(lru_1.hits)
                results['hits_2']               = int(lru_2.hits)
                results['hits_total']           = int(lru_1.hits + lru_2.hits)
                results['hitrate_1']            = float(100*lru_1.hits/lru_1.accesses)
                results['hitrate_2']            = float(100*lru_2.hits/lru_2.accesses)
                results['hitrate_total']        = float(100*total_hits/total_accesses)

                results['bytes_1']              = int(lru_1.accesses * emb_bytes[0])
                results['bytes_2']              = int(lru_2.accesses * emb_bytes[1])
                results['bytes_total']          = int(total_bytes)

                results['bytes_sram_1']         = int(lru_1.hits * emb_bytes[0])
                results['bytes_sram_2']         = int(lru_2.hits * emb_bytes[1])
                results['bytes_sram_total']     = int(total_bytes_sram)

                print(results)

                with open (outfile, 'w') as f:
                    json.dump(results, f)

                experiment_num = experiment_num + 1
        
if __name__ == "__main__":
    main()