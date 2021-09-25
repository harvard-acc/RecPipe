import torch
from torch.optim.lr_scheduler import _LRScheduler

import time
import numpy as np
import json
import math
import sys

import torch


def post_process_queries(args, arrival_rate, num_stages, output_queries, np_y_test_total, stage_batch_sizes):
    preprocess_times = []
    data_times       = []
    inference_times  = []
    fetch_times      = []
    sort_times       = []
    total_times      = []

    stage_times = [ [] for _ in range(num_stages) ]

    for q in output_queries:
        preprocess_times.append ( np.sum ( q.preprocess_times))
        data_times.append       ( np.sum ( q.data_times))
        inference_times.append  ( np.sum ( q.inference_times))
        fetch_times.append      ( np.sum ( q.fetch_times))
        sort_times.append       ( np.sum ( q.sort_times))
        total_times.append      ( q.query_end_time - q.query_start_time)

        for i in range(num_stages):
            stage_times[i].append(get_query_stage_time(q, i))

    # Analyze time breakdown by type of operation
    print("****Average Arrival Rate: ", arrival_rate)

    print("****Total queries: "      , len(output_queries))
    for q in output_queries:
        preprocess_times.append ( np.sum ( q.preprocess_times))
        data_times.append       ( np.sum ( q.data_times))
        inference_times.append  ( np.sum ( q.inference_times))
        fetch_times.append      ( np.sum ( q.fetch_times))
        sort_times.append       ( np.sum ( q.sort_times))
        total_times.append      ( q.query_end_time - q.query_start_time)

    print("****Inference time: "  , np.mean(inference_times  ) , np.percentile(inference_times  , 95 )  )
    print("****Total time: " , np.mean(total_times ) , np.percentile(total_times , 95 )  )

    print("****Preprocess time: " , np.mean(preprocess_times ) , np.percentile(preprocess_times , 95 )  )
    print("****Data time: "       , np.mean(data_times       ) , np.percentile(data_times       , 95 )  )
    print("****Fetch time: "      , np.mean(fetch_times      ) , np.percentile(fetch_times      , 95 )  )

    print("****Sort time: "  , np.mean(sort_times  ) , np.percentile(sort_times  , 95 )  )

    for i in range(num_stages):
        print("****Stage {} average time {}".format(i, np.mean(stage_times[i])))

    print("****model config: ", args.model_configs)
    print("****batch_size: ", stage_batch_sizes)

    ks         = [4, 8, 16, 32, 64]
    dcgs       = [ [] for k in ks]

    if args.data_set == "kaggle":

        qid = 0
        mis_predicts = []
        total_user_items = 0
        incorrect_preds = 0
        for q in output_queries:
            for i, k in enumerate(ks):
                sorted_ids = q.sorted_ids

                y_pred = np.array(q.sorted_scores[:k]).reshape( (-1) )
                y_true = np.array(np_y_test_total[sorted_ids[:k]]).reshape( (-1) )

                dcg_score = ndcg_score(y_true) #over actual scores
                dcgs[i].append(dcg_score)

            y_pred = np.array(q.sorted_scores).reshape( (-1) )
            y_true = np.array(np_y_test_total[sorted_ids]).reshape( (-1) )
            for i in range(len(y_pred)):
                total_user_items += 1
                if int(round(y_pred[i],0)) != y_true[i]:
                    mis_predicts.append(y_pred[i])
                    incorrect_preds += 1

            qid += 1

        for i,k in enumerate(ks):
            print("****DCGs @ {}: {}".format(k, np.mean(dcgs[i])))
        print("****Error : {}".format(incorrect_preds / float(total_user_items) ))

    elif args.data_set == "movielens1m":
        max_samples = 1000
        max_samples = max_samples - 1
        for i, k in enumerate(ks):
            hit_rate = 0
            for q in output_queries:
                sorted_ids = q.sorted_ids
                if max_samples in sorted_ids[:k]:
                    hit_rate += 1

            print("****HR @ {}: {}".format(k, float(hit_rate) / len(output_queries)))

    elif args.data_set == "movielens20m":
        max_samples = 4000
        max_samples = max_samples - 1
        for i, k in enumerate(ks):
            hit_rate = 0
            for q in output_queries:
                sorted_ids = q.sorted_ids
                if max_samples in sorted_ids[:k]:
                    hit_rate += 1

            print("****HR @ {}: {}".format(k, float(hit_rate) / len(output_queries)))


    return

### main loop ###
def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()



def loadGenSleep( sleeptime ):
  if sleeptime > 0.0010:
    time.sleep(sleeptime)
  else:
    startTime = time.time()
    while (time.time() - startTime) < sleeptime:
      continue
  return


class AccelInferenceQuery:
    def __init__(self, query_id = 0, inference_times = None, start_time = None):

        self.query_id        = query_id
        self.inference_times = inference_times

        self.sim_inference_start_times = []
        self.sim_inference_end_times   = []

        self.start_time = start_time
        self.end_time   = None



class InferenceQuery:

    def __init__(self, sorted_scores=None,
                       sorted_ids=None,
                       preprocess_time=None,
                       data_time=None,
                       inference_time=None,
                       fetch_time=None,
                       sort_time=None,
                       query_start_time = None,
                       query_end_time = None,
                       inference_start_time = None,
                       inference_end_time = None,
                       sort_start_time = None,
                       sort_end_time = None,
                       query_id = 0,
                       num_samples = 1,
                       sample_id = 0):

        if sorted_scores is not None:
            self.sorted_scores = sorted_scores
        else:
            self.sorted_scores = None

        if sorted_ids is not None:
            self.sorted_ids       = sorted_ids
        else:
            self.sorted_ids = None

        self.movielens_id = None

        self.preprocess_times = []
        self.data_times       = []
        self.inference_times  = []
        self.fetch_times      = []
        self.sort_times       = []

        if preprocess_time is not None:
            self.preprocess_times.append (preprocess_time)

        if data_time is not None:
            self.data_times.append (data_time)

        if inference_time is not None:
            self.inference_times.append (inference_time)

        if fetch_time is not None:
            self.fetch_times.append (fetch_time)

        if sort_time is not None:
            self.sort_times.append (sort_time)

        self.query_start_time = query_start_time
        self.query_end_time = query_end_time

        self.inference_start_time = []
        self.inference_end_time = []

        if inference_start_time is not None:
            self.inference_start_time.append(inference_start_time)

        if inference_end_time is not None:
            self.inference_end_time.append(inference_end_time)

        self.sort_start_time = []
        self.sort_end_time = []

        if sort_start_time is not None:
            self.sort_start_time.append(sort_start_time)

        if sort_end_time is not None:
            self.sort_end_time.append(sort_end_time)

        # Unique identifier to keep track of queries
        self.query_id     = query_id     # Set by initial loading process
        self.sample_id    = sample_id    # Sample (batch) id of query
        self.num_samples  = num_samples  # Num samples (batches) in query


def ndcg_score(y_pred):
    #https://stackoverflow.com/questions/9468151/how-to-show-that-ndcg-score-is-significant

    dcg = 0
    for i, y in enumerate(y_pred):
        d = y / math.log(2+i, 2)
        dcg += d

    idcg = 0
    for i, y in enumerate(y_pred):
        d = 1. / math.log(2+i, 2)
        idcg += d

    return float(dcg)/idcg

def sort_scrambled_scores(scores, sorted_ids, stage_items):
    scrambled_score_ids = np.argsort(scores)[::-1]
    sorted_ids          = np.array(sorted_ids)[scrambled_score_ids][:stage_items]
    sorted_scores       = scores[scrambled_score_ids][:stage_items] #only need this for last stage
    #sorted_ids         = np.argsort(scores)[::-1]
    #sorted_scores      = scores[sorted_ids]

    return sorted_ids, sorted_scores

def partition_queries(ids, scores, samples_in_batch):
    sample_ids      = []
    sample_scores   = []
    start_sample_id = 0
    for batch_size in samples_in_batch:
        end_sample_id = start_sample_id + batch_size
        sample_ids.append(list(ids[start_sample_id : end_sample_id]))
        sample_scores.append(list(scores[start_sample_id : end_sample_id]))
    return zip(sample_ids, sample_scores)


def split_queries(query_size, batch_size):

    batch_sizes = []
    while query_size > 0:
        sample_size = min(batch_size, query_size)
        batch_sizes.append(sample_size)
        query_size -= sample_size
    return batch_sizes


def get_query_stage_time(query, stage_id):

    stage_time  = query.preprocess_times[stage_id]
    stage_time += query.data_times[stage_id]
    stage_time += query.inference_times[stage_id]
    stage_time += query.fetch_times[stage_id]
    stage_time += query.sort_times[stage_id]

    return stage_time


def count_num_gpus(use_gpus, num_inference_engines):
    num_engines_using_gpu = 0
    for flag, engines in zip(use_gpus, num_inference_engines):
        if flag:
            num_engines_using_gpu += engines

    return num_engines_using_gpu


def is_valid_num_gpus(use_gpus, num_inference_engines):
    c = count_num_gpus(use_gpus, num_inference_engines)
    if  c > 1:
        print("Cannot have more than 1 engine using the GPU", c)
        sys.exit()

    return True


def load_model(args, dlrm):
    use_gpu = args.use_gpu and torch.cuda.is_available()

    print("Loading saved model {}".format(args.load_model))
    if use_gpu:
        if dlrm.ndevices > 1:
            # NOTE: when targeting inference on multiple GPUs,
            # load the model as is on CPU or GPU, with the move
            # to multiple GPUs to be done in parallel_forward
            ld_model = torch.load(args.load_model)
        else:
            # NOTE: when targeting inference on single GPU,
            # note that the call to .to(device) has already happened
            ld_model = torch.load(
                args.load_model,
                map_location=torch.device('cuda')
                # map_location=lambda storage, loc: storage.cuda(0)
            )
    else:
        # when targeting inference on CPU
        ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
    dlrm.load_state_dict(ld_model["state_dict"])
    return dlrm


def dash_separated_ints(value):
    vals = value.split('-')
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value)

    return value


def cli():
    ### import packages ###
    import sys
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser( description="Command line arguments for RecPipe")

    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-sparse-mf-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument( "--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument( "--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument( "--arch-interaction-op", type=str, choices=['dot', 'cat'], default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)

    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7

    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument( "--data-generation", type=str, default="random")  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)

    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)

    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)

    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--use-accel", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=1)

    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)

    parser.add_argument("--model_configs"       , type=str   , default=None)
    parser.add_argument("--recpipe_configs"     , type=str   , default=None)
    parser.add_argument("--accel_configs"       , type=str   , default=None)
    args = parser.parse_args()

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    return args
