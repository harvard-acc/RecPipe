from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
import time
import json
import random

from utils import cli, time_wrap, load_model, InferenceQuery
from utils import get_query_stage_time, split_queries, sort_scrambled_scores, partition_queries
from utils import ndcg_score, is_valid_num_gpus, count_num_gpus, loadGenSleep
from utils import post_process_queries

# Model construction
from models.dlrm_model import DLRM_Net, construct_dlrm_model
from models.neumf import construct_movielens_model, wrap_movielens_inference
import time

from copy import deepcopy

import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

import sklearn.metrics
import sys

from RecPipeAccelSim import accel_thread
from RecPipeAccelSim import run_accel_sim

from multiprocessing import Process, Queue



#####################################################################
# Helper function to build deep learning based recommendation models
#####################################################################
def construct_model(args, data_set_only=False):
    if args.data_set == "kaggle":
        return construct_dlrm_model(args, data_set_only)
    elif "movielens" in args.data_set:
        out = construct_movielens_model(args, data_set_only)
        return out
    else:
        print("Unsupported dataset in model construction!")
        sys.exit()


#####################################################################
# Helper function for inference engines. `inference_thread` defines the process
# for a single CPU/GPU based recommendation inference engine used to rank items
# per stage.
#####################################################################
def inference_thread(args, stage_id, engine_id, qReady, qStart, qInput, qOutput,
                     num_queries, num_threads_per_engine):

    # Build model off of the configuration specified for model architecture
    # parameters
    if args.model_configs is not None:
        with open(args.model_configs, 'r') as f:
            config = json.load(f)
            for key in config:
                type_of = type(getattr(args, key))
                setattr(args, key, type_of(config[key]))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    torch.manual_seed(args.numpy_rand_seed)
    torch.set_num_threads(num_threads_per_engine)

    ###################################################################
    # Constructed inference thread first
    ###################################################################
    use_gpu = args.use_gpu and torch.cuda.is_available()

    dlrm, test_data, test_ld = construct_model(args)

    #Shape of np_X_int and np_X_cat is (length, features). For the full test
    #set length = 3274330 (1million); X_int features is 13 while X_cat features
    #is 26
    if args.data_set == "kaggle":
        np_X_int_test_total = np.array(deepcopy(test_data.X_int))
        np_X_cat_test_total = np.array(deepcopy(test_data.X_cat))

        # Delete test_data and test_loader to save memory space
        del test_data
        del test_ld

        # Warm up caches with full test-set inference
        for i in range(  int((int(len(np_X_cat_test_total)) /1024)-1)):
            start_id = i * 1024
            end_id   = (i+1) * 1024
            dlrm.test_inline(args,
                             np_X_int_test_total[start_id:end_id],
                             np_X_cat_test_total[start_id:end_id])

    elif args.data_set == "movielens1m":
        test_users, test_items = test_data

        # MovieLens1m dataset is preprocessed to rank 1000 items per query.
        # There is at least 1 positive user-item interaction meaning the maximum
        # negative samples are 999.
        max_neg = 999
        max_samples = 1000
        perm = torch.randperm(max_neg)

        # Warm up caches with full test-set inference
        for i, (u,n) in enumerate(zip(test_users,test_items)):

            idx = perm[:max_samples - 1]
            u_samp = u[0][:max_samples]
            i_samp = n[0][torch.cat((idx,torch.tensor([max_neg])),0)]

            u_samp = u_samp.view(-1)
            i_samp = i_samp.view(-1)
            wrap_movielens_inference(args, dlrm, u_samp, i_samp, max_samples)

            if i > 500:
                break

    elif args.data_set == "movielens20m":
        test_users, test_items = test_data

        test_users = test_users[:16000]
        test_items = test_items[:16000]

        # MovieLens1m dataset is preprocessed to rank 4000 items per query.
        # There is at least 1 positive user-item interaction meaning the maximum
        # negative samples are 3999.
        max_neg = 3999
        max_samples = 4000
        perm = torch.randperm(max_neg)

        # Warm up caches with full test-set inference
        for i, (u,n) in enumerate(zip(test_users,test_items)):

            idx = perm[:max_samples - 1]
            u_samp = u[0][:max_samples]
            i_samp = n[0][torch.cat((idx,torch.tensor([max_neg])),0)]

            u_samp = u_samp.view(-1)
            i_samp = i_samp.view(-1)
            wrap_movielens_inference(args, dlrm, u_samp, i_samp, max_samples)

            if i > 64:
                break

    # Send signal to parent process that DLRM model has been constructed and
    # ready to run inferences.
    qReady.put(None)

    # Wait for start signal from parent process
    qStart.get()

    ###################################################################
    # Running inference thread
    ###################################################################
    while True:
        inference_query = qInput.get()

        pre_process_start = time.time()

        if (inference_query == None):
            # If we have received the termination signal infernece process can
            # begin to exit
            #print("[Stage/engine {}/{}] got termination packet".format(stage_id, engine_id))
            qOutput.put(None)
            return

        if args.data_set == "kaggle":

            # Retrieve data
            np_X_int_test = np_X_int_test_total[inference_query.sorted_ids]
            np_X_cat_test = np_X_cat_test_total[inference_query.sorted_ids]

            # Inference
            out = dlrm.test_inline(args, np_X_int_test, np_X_cat_test)

        elif "movielens" in args.data_set:

            # Retrieve data
            test_users, test_items = test_data
            user_id = inference_query.movielens_id
            u_samp = test_users[user_id][0][:len(inference_query.sorted_ids)]
            i_samp = test_items[user_id][0][inference_query.sorted_ids]


            u_samp = u_samp.view(-1)
            i_samp = i_samp.view(-1)

            # Inference
            out = wrap_movielens_inference(args, dlrm, u_samp, i_samp,
                                            len(inference_query.sorted_ids))


        # Unpack output from inference
        scores, preprocess_time, data_times, inference_times, fetch_times = out

        inference_query.sorted_scores = scores

        # Send queries over to next sorting stage
        inference_query.preprocess_times.append(preprocess_time - pre_process_start)
        inference_query.data_times.append(data_times)
        inference_query.inference_times.append(inference_times)
        inference_query.fetch_times.append(fetch_times)

        inference_query.inference_end_time.append(time.time())
        inference_query.inference_start_time.append(pre_process_start)

        qOutput.put(inference_query)

    return


#####################################################################
# Helper function to sort ranked user-item pairs.
# For CPU/GPU based recommendation inference, in addition to inference, sorting
# user-item pairs is a crucial portion of the pipeline. This thread defines the
# process on CPUs. One per recommendation stage.
#####################################################################
def sort_thread(args, sort_id, qReady, qStart, qInput, qOutput, stage_items,
                upstream_infernence_engines, downstream_inference_engines,
                downstream_batch_size):
    # Send signal to parent process that process is ready to proceed
    qReady.put(None)

    # Wait for start signal from parent process
    qStart.get()

    termination_signals_recv = 0

    while True:
        # Receive ranked user-items
        inference_query = qInput.get()

        if inference_query == None:
            termination_signals_recv += 1

            # Wait for all upstream inference engines have compelted processing
            # all queries
            if (termination_signals_recv == upstream_infernence_engines):
                for _ in range(downstream_inference_engines):
                    qOutput.put(None)
                return

            # Wait for next termination signal
            continue

        qkey = inference_query.query_id
        start_time = time.time()

        # Sort ranked user-item pairs
        out = sort_scrambled_scores(inference_query.sorted_scores,
                                    inference_query.sorted_ids, stage_items)
        sorted_ids, sorted_scores = out

        # Based on the scores of user-item pairs, downstream user-item pairs
        # are partitioned befor sending to downstream inference processes
        batch_sizes = split_queries(stage_items, downstream_batch_size)
        id_scores = partition_queries(sorted_ids, sorted_scores, batch_sizes)

        inference_query.num_samples = len(id_scores)
        inference_query.sort_start_time.append(start_time)

        for j, (ids, scores) in enumerate(id_scores):
            q = deepcopy(inference_query)

            # Package query with final sort scores
            q.sorted_scores = np.array(scores, dtype=np.float16)
            q.sorted_ids    = np.array(ids, dtype=np.int32)
            q.sample_id     = j

            end_time = time.time()

            # Add profiling information to query for sort stage
            time_spent = end_time - start_time
            q.sort_times.append(time_spent)
            q.sort_end_time.append(end_time)
            q.query_end_time = end_time

            # Send queries to downstream inference engine
            qOutput.put(q)

    return


#####################################################################
# Main MultiStageRanking (RecPipe) function.
# Based on input parameters (e.g., num_stages, num_threads_per_engine,
# num_inference_engines, stage_batch_sizes, stage_items), we configure
# the RecPipe infrastructure.
#
# Generally, the infrastructure launches separate processes for inference
# engines in each stage as well as combining sorted ID's in 1 sort process
# per stage.
#####################################################################
def MultiStageRanking(args,
                      num_stages,
                      num_queries,
                      num_threads_per_engine,
                      num_inference_engines,
                      stage_batch_sizes,
                      stage_items,
                      nepochs,
                      gpu_flags = None,
                      accel_configs=None,
                      arrival_rates = [5.],
                      use_accel=False,
                      ):

    random.seed(args.numpy_rand_seed)
    np.random.seed(args.numpy_rand_seed)
    torch.manual_seed(args.numpy_rand_seed)

    # Using GPUs
    if gpu_flags is not None:
        # Check valid number of GPUs
        valid_num_gpus = is_valid_num_gpus(gpu_flags, num_inference_engines)

    # Calculate the number of inference engines using GPUs (min 0, max 1)
    num_engines_using_gpu = count_num_gpus(gpu_flags, num_inference_engines)

    #Shape of np_X_int and np_X_cat is (length, features). For the full test
    #set length = 3274330 (1million); X_int features is 13 while X_cat features
    #is 26
    if args.data_set == "kaggle":
        _, test_data, _ = construct_model(args, data_set_only=True)
        np_y_test_total = np.array(deepcopy(test_data.y))
        total_test_size = np.array(test_data.X_int).shape[0]
        ids             = list(range(total_test_size)[:stage_items[0]])

        del test_data # Saving memory as process memory will be copied to children

    # ##############
    # Main processes for MultiStageRanking(MSR) infrastructure
    # ##############

    # Inference and sort engines
    MSRInferenceEngines = [ [] for _ in range(num_stages) ]
    MSRSortEngines      = []

    # Queues to start MSRInference processes
    MSRInferenceReadyQueues  = [ [] for _ in range(num_stages) ]
    MSRInferenceStartQueues  = [ [] for _ in range(num_stages) ]

    # Queues to start MSRSort processes
    MSRSortReadyQueues  = [ ]
    MSRSortStartQueues  = [ ]

    # Queues to transfer inputs between MSRInference-MSRSort-MSRInference
    # processes.
    MSRInferenceInputQueue = [ Queue() for _ in range(num_stages) ]
    MSRSortInputQueue      = [ Queue() for _ in range(num_stages) ]
    MSROutputQueue         = Queue()

    # Model configurations specifying which models are used per stage

    model_configs = args.model_configs.split(",")
    # ##############
    # Instantiate multi-stage inference processes
    # ##############
    for i in range(num_stages):

        # Create the specified number of inference engines
        for j in range(num_inference_engines[i]):
            qReady  = Queue()
            qStart  = Queue()

            MSRInferenceReadyQueues[i].append(qReady)
            MSRInferenceStartQueues[i].append(qStart)

            engine_args                      = deepcopy(args)
            engine_args.mini_batch_size      = stage_batch_sizes[i]
            engine_args.test_mini_batch_size = stage_batch_sizes[i]
            engine_args.model_configs        = model_configs[i]
            engine_args.use_gpu              = gpu_flags[i]

            # MSRInferenceEngines and MSRSort process arguments if not using
            # accelerator
            if use_accel == False:
                pargs = (engine_args, i, j, qReady, qStart, MSRInferenceInputQueue[i],
                        MSRSortInputQueue[i], num_queries,
                        num_threads_per_engine[i])

            # MSRInferenceEngines and MSRSort process arguments if using
            # accelerator
            else:
                engine_args.accel_configs = accel_configs[i]
                if i != num_stages - 1:
                    pargs = (engine_args, i, j, qReady, qStart, MSRInferenceInputQueue[i],
                            MSRInferenceInputQueue[i+1], num_queries,
                            num_threads_per_engine[i])
                else:
                    pargs = (engine_args, i, j, qReady, qStart, MSRInferenceInputQueue[i],
                            MSROutputQueue, num_queries,
                            num_threads_per_engine[i],)

            if use_accel == True:
                # Inference engine uses accelerator thread
                p = Process( target = accel_thread, args = pargs)
            else:
                # Inference engine uses CPU/GPU inference thread
                p = Process( target = inference_thread, args = pargs)

            MSRInferenceEngines[i].append(p)

        # If we are not using the accelerator (RecPipeAccel) sorting ranked
        # items across stages must be done on CPU/GPU based systems. Here we
        # launch one sorting process per stage.
        if use_accel == False:
            sortReady = Queue()
            sortStart = Queue()
            MSRSortReadyQueues.append(sortReady)
            MSRSortStartQueues.append(sortStart)
            if i != num_stages - 1:
                pargs = (args, i, sortReady, sortStart, MSRSortInputQueue[i],
                        MSRInferenceInputQueue[i+1], stage_items[i+1],
                        num_inference_engines[i], num_inference_engines[i+1],
                        stage_batch_sizes[i+1])
            else:
                # Set stage_items and batch_sizes to be equivalent for last stage
                pargs = (args, i, sortReady, sortStart, MSRSortInputQueue[i],
                        MSROutputQueue, stage_items[i], num_inference_engines[i],
                        1, stage_items[i])
            p = Process( target = sort_thread, args=pargs)
            MSRSortEngines.append(p)

    #################################################################
    # Launch multi-stage inference processes
    #################################################################
    for i in range(num_stages):
        if use_accel == False:
            MSRSortEngines[i].start()

        for j in range(num_inference_engines[i]):
            MSRInferenceEngines[i][j].start()

    for i in range(num_stages):
        if use_accel == False:
            MSRSortReadyQueues[i].get()

        for j in range(num_inference_engines[i]):
            MSRInferenceReadyQueues[i][j].get()

    for i in reversed(range(num_stages)):
        if use_accel == False:
            MSRSortStartQueues[i].put(None)

        for j in range(num_inference_engines[i]):
            MSRInferenceStartQueues[i][j].put(None)
            print("Launched inference engine {}/{}".format(i, j))

    #################################################################
    # Add input to drive the multi-stage ranking pipeline
    #################################################################
    # Sweep arrival rates based on RecPipe experiment sweeps' configurations
    for arrival_rate in arrival_rates:

        # Epochs can be used to model rnadom seeds for averaging results over
        # many runs
        for epoch in range(nepochs):

            print("===================================================")
            print("****Epoch ", epoch)
            sys.stdout.flush()
            exp_start_time = time.time()
            total_queries = 0
            samples_in_batch = split_queries(stage_items[0], stage_batch_sizes[0])

            for i in range(num_queries):

                # Criteo Kaggle data set
                if args.data_set == "kaggle":
                    start_sample_id  = 0
                    # Support to split queries into multiple batches. For
                    # RecPipe we do not consider this additional dimension of
                    # optimization.
                    for j, batch_size in enumerate(samples_in_batch):
                        end_sample_id = start_sample_id + batch_size
                        sample_ids = list(ids[start_sample_id : end_sample_id])

                        # InferenceQuery
                        q = InferenceQuery(sorted_ids=np.array(sample_ids,
                                           np.int32), query_id = i,
                                num_samples=len(samples_in_batch), sample_id=j)

                        q.query_start_time = time.time()
                        MSRInferenceInputQueue[0].put(q)
                        start_sample_id += batch_size

                    # Update ids for next query so we use new ids each time
                    ids = list(map(lambda x: (x + stage_items[0]) % total_test_size, ids))

                # MovieLens 1 million dataset
                elif args.data_set == "movielens1m":

                    # MovieLens1M dataset is pre-processed such that each query
                    # has at most 1000 possible movies to rank
                    max_samples = 1000

                    # at least 1 sample must be positive
                    max_neg = max_samples - 1

                    # Dataset has 2800 users
                    max_users = 2800
                    assert(max_samples >= stage_items[0])

                    # Generate IDs and make sure last positive sample is in
                    # list
                    perm = torch.randperm(max_neg).numpy()[:stage_items[0]]
                    sample_ids = np.append(perm ,[max_neg])

                    q = InferenceQuery(sorted_ids=np.array(sample_ids, np.int32),
                                       query_id = i, num_samples=1, sample_id=0)
                    q.movielens_id = i % max_users
                    q.query_start_time = time.time()

                    MSRInferenceInputQueue[0].put(q)

                # MovieLens 20 million dataset
                elif args.data_set == "movielens20m":
                    # MovieLens1M dataset is pre-processed such that each query
                    # has at most 4000 possible movies to rank
                    max_samples = 4000 # TODO: Lets not hardcode this?

                    # at least 1 sample must be positive
                    max_neg = max_samples - 1

                    # Dataset has 5000 users
                    max_users = 5000
                    assert(max_samples >= stage_items[0])

                    # Generate IDs and make sure last positive sample is in
                    # list
                    perm = torch.randperm(max_neg).numpy()[:stage_items[0]]
                    sample_ids = np.append(perm ,[max_neg])

                    q = InferenceQuery(sorted_ids=np.array(sample_ids, np.int32),
                                       query_id = i, num_samples=1, sample_id=0)
                    q.movielens_id = i % max_users
                    q.query_start_time = time.time()

                    MSRInferenceInputQueue[0].put(q)

                else:
                    print("Unsupported dataset while generated queries!")
                    sys.exit()

                # Modeling poisson arrival rate
                if use_accel == False:
                    # For accelerator we handle poisson arrival rates
                    # separately and instead use MultiStageRanking to generate
                    # per-query inference times. See
                    # run_AccelMultiStageRankingSweep and RecPipeMain
                    arrival_time = np.random.poisson(lam = arrival_rate, size = 1)
                    loadGenSleep( arrival_time / 1000.   )
                total_queries += 1
            sys.stdout.flush()

            #################################################################
            #Handle output
            output_queries = []
            while len(output_queries) != total_queries:
                if MSROutputQueue.qsize() != 0:
                    q = MSROutputQueue.get()
                    if q is not None:
                        output_queries.append(q)

            exp_end_time = time.time()

            #################################################################
            # Analyze time breakdown by type of operation
            inference_times = []
            if use_accel:
                for q in output_queries:
                    inference_times.append( q.inference_times )

            exp_time = (exp_end_time - exp_start_time)
            print("****Total experiment time (s): ",  exp_time )

            if use_accel == False:
                # Criteo Kaggle dataset
                if args.data_set == "kaggle":
                    post_process_queries(args, arrival_rate, num_stages,
                                         output_queries, np_y_test_total,
                                         stage_batch_sizes)

                # MovieLens1m and MovieLens20m datasets
                else:
                    post_process_queries(args, arrival_rate, num_stages,
                                         output_queries, None,
                                         stage_batch_sizes)

                print("****GPU Flags", gpu_flags)

            print("===================================================")

            time.sleep(2)

    #################################################################
    # Tear down MultiStageRanking process and end RecPipe experiments
    #################################################################
    for i in range(num_stages):
        for _ in range(num_inference_engines[i]):
            if use_accel:
                MSRInferenceInputQueue[i].put(None)
            else:
                MSRInferenceInputQueue[0].put(None)

    #Clean up inference engines once they are finished
    for i in range(num_stages):
        if use_accel == False:
            MSRSortEngines[i].join()

        for j in range(num_inference_engines[i]):
            MSRInferenceEngines[i][j].join()

    while MSROutputQueue.qsize() != 0:
        q = MSROutputQueue.get()
        if q != None:
            print("Warning: queries left in output queue")

    if use_accel:
        return inference_times
    else:
        return 0

