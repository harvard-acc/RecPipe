from RecPipe import MultiStageRanking
from RecPipeAccelSim import run_accel_sim
from utils import cli
import math
import json
import os

from copy import deepcopy

#####################################################################
# Sweep through all experiment configurations for RecPipe experiments
# on CPU and CPU-GPU based systems.
#####################################################################
def run_MultiStageRankingSweep(args, num_queries, nepochs, num_stages,
                               max_cpu_engines,
                               model_configs,
                               arrival_rates,
                               num_threads_per_engine,
                               stage_batch_sizes,
                               use_gpus
                               ):

    # Iterate through model configurations
    for k, model in enumerate(model_configs):
        exp_args = deepcopy(args)
        setattr(exp_args, 'model_configs', model)

        # Iterate through number of items ranked per stage
        for stage_batch_size in stage_batch_sizes:

            # Iterate through CPU/GPU mapping. Assumes GPU is used for at most
            # 1 stage
            for use_gpu in use_gpus:
                engines = [ 0 for _ in range(len(use_gpu))]

                for i, g in enumerate(use_gpu):
                    if g:
                        engines[i] = int(1)
                    else:
                        num_gpus = sum(use_gpu)
                        engines[i] = int(max_cpu_engines / (len(use_gpu)-num_gpus))

                # Run MultiStageRanking experiment
                error = MultiStageRanking(exp_args, num_stages, num_queries,
                        num_threads_per_engine, engines, stage_batch_size,
                        stage_batch_size, nepochs, gpu_flags = use_gpu,
                        arrival_rates = arrival_rates,)
        _ = os.system('date')

    return

#####################################################################
# Sweep through all experiment configurations for RecPipe experiments
# on accelerator based systems (RecPipeAccel).
#####################################################################
def run_AccelMultiStageRankingSweep(args, num_queries, nepochs, num_stages,
                               model_configs, accel_configs,
                               arrival_rates,
                               num_threads_per_engine,
                               stage_batch_sizes,
                               num_accel_engines,
                               ):

    # Iterate through model configurations
    for k, model in enumerate(model_configs):

        # Iterate through accelerator configurations
        for aid, accel in enumerate(accel_configs):
            exp_args = deepcopy(args)

            # Set model configuration for each stage in recommendation pipeline
            setattr(exp_args, 'model_configs', model)

            # Iterate through number of items ranked per stage
            for stage_batch_size in stage_batch_sizes:

                # Iterate through number of accelerator engines across stages
                for num_accel_engine in num_accel_engines:
                    # For accelerator, disable GPUs for all engines
                    use_gpu = [ False for _ in num_accel_engine]

                    # For accelerator experiments we first run
                    # MultiStageRanking offline to gather per-query inference
                    # times (inf_times) by invoking accelerator model
                    # (RecPipeAccel).
                    inf_times = MultiStageRanking(exp_args,
                                                  num_stages,
                                                  num_queries,
                                                  num_threads_per_engine,
                                                  num_accel_engine,
                                                  stage_batch_size,
                                                  stage_batch_size,
                                                  1,
                                                  gpu_flags = use_gpu,
                                                  accel_configs = accel,
                                                  arrival_rates = [2],
                                                  use_accel=True,
                                                  )

                    # Given per-query inference times (inf_times) we know
                    # emulate at-scale execution by running simulation process
                    # per accelerator engine
                    run_accel_sim( exp_args,
                                   num_stages,
                                   num_queries,
                                   num_threads_per_engine,
                                   num_accel_engine,
                                   stage_batch_size,
                                   stage_batch_size,
                                   nepochs, #E
                                   inf_times,
                                   accel,
                                   arrival_rates = arrival_rates,
                                   )
            _ = os.system('date')

    return


def accel_sweep(args):
    stage_batch_sizes = [
                  [4096],
                  [6*1024],
                  [8*1024],
                  [10*1024],
                  [12*1024],
                  [14*1024],
                  [16*1024],
                  [18*1024],
                  [20*1024],
                  ]

    stage_batch_sizes = [
                          [4*1024  , 512] ,
                          [6*1024  , 512] ,
                          [8*1024  , 512] ,
                          [10*1024 , 512] ,
                          [12*1024 , 512] ,
                          [14*1024 , 512] ,
                          [16*1024 , 512] ,
                          [18*1024 , 512] ,
                          [20*1024 , 512] ,
                          [24*1024 , 512] ,
                          [26*1024 , 512] ,
                          [28*1024 , 512] ,
                        ]


def accelMovieLensSweep(args):

    #######################################################
    # Single stage system
    #######################################################
    num_queries     = 1000 # Number of input queries to rank
    nepochs         = 1

    #arrival_rates  = [ 25, 10, 2., 1., 0.5 ]
    arrival_rates  = [  1. ]

    use_gpus       = [ [False], ]

    num_stages = 1  # Number of multi-ranking stages
    # Always keep this at one given we are only considering single-threaded
    # inference
    num_threads_per_engine = [1] # Number of threads per CPU inference Process

    ## Only consider smallest model for accelerator simulation
    model_configs = [ 'configs/model_configs/movie20m_large.json', ]

    stage_batch_sizes = [ [4000], ]
    exp = 0

    for accel_size in [1, 2, 4, 8]:
        accel_str = 'configs/accel_configs/hardconf_{}.json'.format(accel_size)
        accel_configs = [ [accel_str] ]
        num_accel_engines       = [ [accel_size], ]
        exp += 1
        #run_AccelMultiStageRankingSweep(args, num_queries, nepochs, num_stages,
        #                           model_configs, accel_configs,
        #                           arrival_rates,
        #                           num_threads_per_engine, stage_batch_sizes,
        #                           num_accel_engines,
        #                           )

    num_stages = 2  # Number of multi-ranking stages
    # Always keep this at one given we are only considering single-threaded
    # inference
    num_threads_per_engine = [1, 1] # Number of threads per CPU inference Process

    # Only consider smallest model for accelerator simulation
    model_configs = [ 'configs/model_configs/movie20m_small.json,configs/model_configs/movie20m_large.json', ]
    stage_batch_sizes = [ [4000, 400], ]

    for front in [1, 2, 4, 8]:
        for back in [1, 2, 4, 8]:
            front_str = 'configs/accel_configs/hardconf_{}.json'.format(front*2)
            back_str = 'configs/accel_configs/hardconf_{}.json'.format(back*2)
            accel_configs = [ [front_str, back_str] ]

            num_accel_engines = [ [front, back] ]
            exp += 1
            #run_AccelMultiStageRankingSweep(args, num_queries, nepochs, num_stages,
            #                           model_configs, accel_configs,
            #                           arrival_rates,
            #                           num_threads_per_engine, stage_batch_sizes,
            #                           num_accel_engines,
            #                           )

    num_stages = 3  # Number of multi-ranking stages
    # Always keep this at one given we are only considering single-threaded
    # inference
    num_threads_per_engine = [1, 1, 1] # Number of threads per CPU inference Process

    # Only consider smallest model for accelerator simulation
    model_configs     = [ 'configs/model_configs/movie20m_small.json,configs/model_configs/movie20m_medium.json,configs/model_configs/movie20m_large.json', ]
    stage_batch_sizes = [ [4000, 400, 250], ]

    units = 16

    accel_tops =[ (2, 1), (4, 1), (4, 2),
                     (8, 1), (8, 2), (8, 4),
                     (16, 1), (16, 2), (16, 4), (16, 8)]

    for front, nf in accel_tops:
        for middle, nm in accel_tops:
            for back, nb in accel_tops:
                size = (16/front) * nf + (16/middle) * nm + (16/back) * nb
                if size != 16:
                    continue

                front_str  = 'configs/accel_configs/hardconf_{}.json'.format(front)
                middle_str = 'configs/accel_configs/hardconf_{}.json'.format(middle)
                back_str   = 'configs/accel_configs/hardconf_{}.json'.format(back)
                accel_configs = [ [front_str, middle_str, back_str] ]

                num_accel_engines = [ [nf, nm, nb] ]
                exp += 1

                run_AccelMultiStageRankingSweep(args, num_queries, nepochs, num_stages,
                                           model_configs, accel_configs,
                                           arrival_rates,
                                           num_threads_per_engine, stage_batch_sizes,
                                           num_accel_engines,
                                           )
    return


#####################################################################
# Helper functions to convert GPU configurations from string format to list of
# booleans
#####################################################################
def gpu_strs_to_bools(string):
    gpus = []
    for x in string:
        if "False" == x:
            gpus.append(False)
        elif "True" == x:
            gpus.append(True)
        else:
            print("Json configuration error: False or True for gpus")
    return gpus

def listify_gpus(string):
    list_gpus = list(map(lambda x: gpu_strs_to_bools(x.split(",")), string))
    return list_gpus

#####################################################################
# Helper function to convert items ranked per stage configurations from string
# format to list of integers
#####################################################################
def batch_strs_to_int(string):
    batch = []
    for x in string:
        batch.append(int(x))
    return batch

def listify_batch_sizes(string):
    list_batch = list(map(lambda x: batch_strs_to_int(x.split(",")), string))
    return list_batch

#####################################################################
# Helper function to convert accelerator configurations from string
# format to list of integers
#####################################################################
def listify_accel_configs(string):
    list_accel = list(map(lambda x: batch_strs_to_int(x.split(",")), string))
    return list_accel

#####################################################################
# Helper function to generate all accelerator possible configurations
# based on splitting a monotithic accelerator into multiple sub-accelerator
# nodes in order to exploit concurrency in parallel queries
#####################################################################
def add_accel_config(sweep_accel_configs, config):
    if config in sweep_accel_configs:
        return sweep_accel_configs
    else:
        sweep_accel_configs.append(config)
        return sweep_accel_configs

def gen_sweep_accel_configs(num_stages):
    sweep_accel_configs = []

    # All possible accelerator sizes for stages.
    # (size, number) where
    # size denotes size of accel. (1 is largest, 4 is 1/4 size, 16 is 1/16 size)
    # number denotes the number of accel. nodes
    accel_tops = [ (1, 1),
                   (4,1), (4,2), (4,4),
                   (16,1), (16,2), (16,4), (16,8), (16,16),
                   (64,1), (64,2), (64,4), (64,8), (64,16), (64,32), (64,64)
                 ]

    # Single stage configurations
    if num_stages == 1:
        for front, nf in accel_tops:
            size = (64/front) * nf
            if size <= 64:
                sweep_accel_configs = add_accel_config(sweep_accel_configs, [front])

    # 2 stage configurations
    elif num_stages == 2:
        for front, nf in accel_tops:
            for back, nb in accel_tops:
                size = (64/front) * nf + (64/back) * nb
                if size <= 64:
                    sweep_accel_configs = add_accel_config(
                                                sweep_accel_configs,
                                                [front, back])

    # 3 stage configurations
    elif num_stages == 3:
        for front, nf in accel_tops:
            for middle, nm in accel_tops:
                for back, nb in accel_tops:
                    size = (64/front) * nf + (64/middle) * nm + (64/back) * nb
                    if size <= 64:
                        sweep_accel_configs = add_accel_config(
                                                sweep_accel_configs,
                                                [front, middle, back])
    else:
        print("Error. Unsupport number of accelerator stages")
        sys.exit()
    return sweep_accel_configs


if __name__ == "__main__":
    # Parse command line options
    args = cli()

    if args.recpipe_configs is not None:
        with open(args.recpipe_configs, 'r') as f:
            config = json.load(f)

            # Parse RecPipe experiment configurations
            num_queries            = config['num_queries']
            nepochs                = config['nepochs']
            arrival_rates          = config['arrival_rates']
            max_cpu_engines        = config['max_cpu_engines']
            num_stages             = config['num_stages']
            model_configs          = config['model_configs']
            use_gpus               = listify_gpus(config['use_gpus'])
            num_threads_per_engine = config['num_threads_per_engine']
            stage_batch_sizes      = listify_batch_sizes(config['stage_batch_sizes'])

            # Uncomment if you want to print RecPipe experiment configurations
            #print(num_queries)
            #print(nepochs)
            #print(arrival_rates)
            #print(max_cpu_engines)
            #print(num_stages)
            #print(model_configs)
            #print(use_gpus)
            #print(num_threads_per_engine)
            #print(stage_batch_sizes)

    # RecPipe accelerator based experiments then use
    # run_AccelMultiStageRankingSweep
    if args.use_accel:

        # Sweep all possible accelerator configurations
        if config['accel_configs'] == "all":
            sweep_accel_configs = gen_sweep_accel_configs(num_stages)

        # Sweep configurations specified in accelerator json configs
        else:
            sweep_accel_configs = listify_accel_configs(config['accel_configs'])

        # Sweep accelerator configurations
        for exp_accel_config in sweep_accel_configs:
            accel_configs     = []
            num_accel_engines = []
            for accel_size in exp_accel_config:
                accel_str = 'configs/accel_configs/hardconf_{}.json'.format(accel_size)
                accel_configs.append(accel_str)
                num_accel_engines.append(int(math.floor(accel_size / num_stages)))

            accel_configs     = [ accel_configs ]
            num_accel_engines = [ num_accel_engines ]

            # Run accelerator (RecPipeAccel) sweeps
            run_AccelMultiStageRankingSweep(args, num_queries, nepochs, num_stages,
                                       model_configs, accel_configs,
                                       arrival_rates,
                                       num_threads_per_engine, stage_batch_sizes,
                                       num_accel_engines
                                       )

    # Else --- RecPipe CPU/GPU based experiments then use
    # run_MultiStageRankingSweep
    else:
        run_MultiStageRankingSweep(args, num_queries, nepochs, num_stages,
                                   max_cpu_engines,
                                   model_configs, arrival_rates,
                                   num_threads_per_engine, stage_batch_sizes,
                                   use_gpus
                                   )

    #accelMovieLensSweep(args)
