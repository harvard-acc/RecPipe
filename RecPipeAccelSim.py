
import json

import numpy as np
import torch

import models.dlrm_data_pytorch as dp

from copy import deepcopy
import sys
import time

from utils import loadGenSleep
from utils import AccelInferenceQuery
import numpy as np
import random

import RecPipeAccelModel
from multiprocessing import Process, Queue


##########################################################################
# Accelerator simulation thread to measure the at-scale implications of
# different accelerator topologies. Input queries to accel_sim_thread
# are per-query and per-stage inference times.
##########################################################################
def accel_sim_thread(args, stage_id, engine_id, qInput, qOutput,
                    num_queries, num_threads_per_engine):

    query_count = 0

    time_total  = 0
    time_active = 0
    time_active_start = 0

    ###################################################################
    # Running inference thread
    ###################################################################
    while True:
        # Receive inference query
        inference_query = qInput.get()

        if query_count == 256:
            time_total = time.time()

        # Done processing all queries
        if (inference_query == None):
            time_total = (time.time() - time_total)
            print("Accel {} util: {}".format(stage_id, time_active / float(time_total) * 100))
            return

        # Model/emulate inference time per query
        inference_start = time.time()
        time.sleep(inference_query.inference_times[stage_id])
        inference_end = time.time()

        if query_count >= 256:
            time_active += (inference_end - inference_start)

        # Send query to next set of modeled/emulated accelerator
        inference_query.sim_inference_start_times.append(inference_start)
        inference_query.sim_inference_end_times.append(inference_end)

        inference_query.end_time = inference_end
        qOutput.put(inference_query)

        query_count += 1

    return


##########################################################################
# Main simulation infrastructure for multi-stage recommendation accelerators.
# run_accel_sim sets up RecPipeAccel for each stage of the multi-stage pipeline
##########################################################################
def run_accel_sim( args,
                   num_stages,
                   num_queries,
                   num_threads_per_engine,
                   num_inference_engines,
                   stage_batch_sizes,
                   stage_items,
                   nepochs,
                   inference_query_runtimes,
                   accel_configs,
                   arrival_rates = [5.],
                   ):

    random.seed(args.numpy_rand_seed)
    np.random.seed(args.numpy_rand_seed)

    AccelInferenceEngines = [ [] for _ in range(num_stages) ]

    AccelInferenceInputQueue = [ Queue() for _ in range(num_stages) ]
    AccelOutputQueue         = Queue()

    # Model configuration for per-stage
    model_configs = args.model_configs.split(",")

    #################################################################
    # Instantiate multi-stage inference processes
    #################################################################
    for i in range(num_stages):

        # Create the specified number of inference engines
        for j in range(num_inference_engines[i]):
            engine_args                      = deepcopy(args)
            engine_args.mini_batch_size      = stage_batch_sizes[i]
            engine_args.test_mini_batch_size = stage_batch_sizes[i]
            engine_args.model_configs        = model_configs[i]
            engine_args.accel_configs        = accel_configs[i]

            # Process arguments for each stage
            if i != num_stages - 1:
                pargs = (engine_args, i, j, AccelInferenceInputQueue[i],
                        AccelInferenceInputQueue[i+1], num_queries,
                        num_threads_per_engine[i])
            else:
                pargs = (engine_args, i, j, AccelInferenceInputQueue[i],
                        AccelOutputQueue, num_queries,
                        num_threads_per_engine[i])

            # Launch accelerator thread
            p = Process( target = accel_sim_thread, args = pargs)

            AccelInferenceEngines[i].append(p)

    #################################################################
    # Launch multi-stage inference processes
    #################################################################
    for i in range(num_stages):
        for j in range(num_inference_engines[i]):
            AccelInferenceEngines[i][j].start()

    #################################################################
    # Add input to drive the multi-stage ranking pipeline
    # Iterate through all arrival rates based on RecPipe experiment sweep
    # configuration
    #################################################################
    for arrival_rate in arrival_rates:
        #Epochs can be used to model rnadom seeds for averaging results over many
        #runs
        for epoch in range(nepochs):
            print("===================================================")
            print("****Epoch ", epoch)
            exp_start_time = time.time()
            total_queries = 0

            for i in range(num_queries):
                q = AccelInferenceQuery( query_id=i,
                                         inference_times=inference_query_runtimes[i],
                                         start_time = time.time() )

                AccelInferenceInputQueue[0].put(q)

                # Modeling poisson arrival rate
                arrival_time = np.random.poisson(lam = arrival_rate, size = 1)
                loadGenSleep( arrival_time / 1000. )
                total_queries += 1

            sys.stdout.flush()

            #################################################################
            #Handle output
            output_queries = []
            while len(output_queries) != total_queries:
                if AccelOutputQueue.qsize() != 0:
                    q = AccelOutputQueue.get()
                    if q is not None:
                        output_queries.append(q)

            exp_end_time = time.time()

            #################################################################
            # Analyze time breakdown by type of operation
            inference_times      = []
            tail_inference_times = []
            for q in output_queries:
                inference_times.append( np.sum(q.inference_times) )
                tail_inference_times.append(q.end_time - q.start_time)
                #print("Inference stages: ", q.inference_times)
                #print("Tail inference times: ", tail_inference_times[-1])

            exp_time = (exp_end_time - exp_start_time)

            # Final experiment statistics
            print("****Total experiment time (s): " , exp_time )
            print("****Infer mean time (s): "       , np.mean(inference_times) )
            print("****Infer tail time (s): "       , np.percentile(inference_times      , 95) )
            print("****Total mean time (s): "       , np.mean(tail_inference_times) )
            print("****Total tail time (s): "       , np.percentile(tail_inference_times , 95) )
            print("****Arrival rate: "              , arrival_rate)
            print("****model config: "              , args.model_configs)
            print("****batch_size: "                , stage_batch_sizes)
            print("****accel config: "              , accel_configs)
            print("****accel counts: "              , num_inference_engines)

            print("===================================================")
            # Some cool off period from RecPipe infrastructure before next
            # experiment
            time.sleep(2)

    # Once all queries and experiments are finished send done signal
    for i in range(num_stages):
        for _ in range(num_inference_engines[i]):
            AccelInferenceInputQueue[i].put(None)

    #Clean up inference engines once they are finished
    for i in range(num_stages):
        for j in range(num_inference_engines[i]):
            AccelInferenceEngines[i][j].join()

    while AccelOutputQueue.qsize() != 0:
        q = AccelOutputQueue.get()
        if q != None:
            print("Warning: queries left in output queue")

    return


##########################################################################
# Helper function to build dataset for accelerator experiments
##########################################################################
def construct_dataset(args, ):
    use_gpu = False

    device = torch.device("cpu")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if (args.data_generation == "dataset"):

        test_data, test_ld = dp.make_criteo_data_and_loaders(args)

        ln_emb = test_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = test_data.m_den
        ln_bot[0] = m_den
    else:
        print("Error need a dataset")
        sys.exit()

    print("Created dataset")
    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )

    if m_spa != m_den_out:
        sys.exit(
            "ERROR: arch-sparse-feature-size "
            + str(m_spa)
            + " does not match last dim of bottom mlp "
            + str(m_den_out)
        )

    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    return test_data, test_ld


##########################################################################
# Accelerator thread to measure the per-inference query times
# based on different accelerator (RecPipeAccel) topologies
##########################################################################
def accel_thread(args, stage_id, engine_id, qReady, qStart, qInput, qOutput,
                     num_queries, num_threads_per_engine,):

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

    ### accelerator modeling setup ###
    accelerator = RecPipeAccelModel.Accelerator(args.model_configs,
                                          args.accel_configs,
                                          args.data_set)

    ###################################################################
    # Constructed inference thread first
    ###################################################################
    use_gpu = args.use_gpu and torch.cuda.is_available()

    if args.data_set == "kaggle":
        test_data, test_ld = construct_dataset(args)

        #Shape of np_X_int and np_X_cat is (length, features). For the full test
        #set length = 3274330 (1million); X_int features is 13 while X_cat features
        #is 26
        np_X_int_test_total = np.array(deepcopy(test_data.X_int))
        np_X_cat_test_total = np.array(deepcopy(test_data.X_cat))
        np_y                = np.array(deepcopy(test_data.y))

        # Delete test_data and test_ld for memory saving
        del test_data
        del test_ld

        # Warm up caches with full test-set inference
        for i in range(  int((int(len(np_X_cat_test_total)) /1024)-1)):
            start_id = i * 1024
            end_id = (i+1) * 1024

            # accelerator inference to warm up caches
            # Input continuous features:   np_X_int_test_total[start_id:end_id]
            # Input cateogorical features: np_X_cat_test_total[start_id:end_id]
            accelerator.warm_cache(np_X_cat_test_total[start_id:end_id])

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

        if inference_query == None:
            # If we have received the termination signal infernece process can
            # begin to exit
            #print("[Stage/engine {}/{}] got termination packet".format(stage_id, engine_id))
            qOutput.put(None)
            return

        pre_process_start = time.time()

        # Retrieve sorted IDs from query for continuous and categorical features
        #np_X_int_test = np_X_int_test_total[inference_query.sorted_ids]
        if args.data_set == "kaggle":
            np_X_cat_test = np_X_cat_test_total[inference_query.sorted_ids]
        else:
            np_X_cat_test = []
        data_time = time.time()

        # Inference
        # Return out -> scores (placeholder), preprocess_time, data_times, inference_times, fetch_times
        # preprocess_time = Actual timestamp of preocessping data (Not really important)
        # data_times      = Time to read inputs from data (Not really important)
        # inference_times = Time for actual inference  (Very important)
        # fetch_times     = Time to fetch data from accelerator (Important for return of data from PCIe)
        items_per_query = args.mini_batch_size
        top_n = 0
        (scores,
         preprocess_time,
         data_times,
         inference_times,
         fetch_times) = accelerator.run_query(items_per_query, top_n, np_X_cat_test)

        sim_time = time.time()

        loadGenSleep(inference_times) # Sleep to model accelerator tradeoff

        end_time = time.time()
        #print("Time to read data: ", data_time - pre_process_start)
        #print("Simulation time: ", sim_time - data_time)

        inference_query.query_end_time = end_time
        inference_query.inference_end_time.append(end_time)
        inference_query.inference_start_time.append(pre_process_start)

        # Convert inference_times into
        if args.data_set == "kaggle":
            inference_query.sorted_scores = np_y[inference_query.sorted_ids]
        else:
            inference_query.sorted_scores = []

        # Send queries over to next sorting stage
        inference_query.preprocess_times.append(0)
        inference_query.data_times.append(0)
        inference_query.inference_times.append(inference_times)
        inference_query.fetch_times.append(0)

        inference_query.sort_times.append(0)
        dump_query_time = time.time()

        qOutput.put(inference_query)

    return

