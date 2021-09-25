from copy import deepcopy
import sys


def pareto_frontier_z(Xs, Ys, Zs, maxX = True, maxY=True):
    myList = sorted([ [Xs[i], Ys[i], Zs[i]] for i in range(len(Xs))], reverse=maxX)

    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY and (round(pair[1], 4) > round(p_front[-1][1], 4)):
            if maxX and (round(pair[0], 4) < round(p_front[-1][0], 4)):
                p_front.append(pair)
            elif (not maxX) and (round(pair[0], 4) > round(p_front[-1][0], 4)):
                p_front.append(pair)
        elif (not maxY) and (round(pair[1], 4) < round(p_front[-1][1], 4)):
            if maxX and (round(pair[0], 4) < round(p_front[-1][0], 4)):
                p_front.append(pair)
            elif (not maxX) and (round(pair[0], 4) > round(p_front[-1][0], 4)):
                p_front.append(pair)
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    p_frontZ = [pair[2] for pair in p_front]

    return p_frontX, p_frontY, p_frontZ


def model_to_rm(string):
    if 'dim_4_0' in string:
        return 'RM-small'
    if 'dim_16' in string:
        return 'RM-med'
    if 'dim_32' in string:
        return 'RM-large'


def main():
    breaker = "================================"

    lines  = []
    files  = ['logs/example_logs/figure7_8_kaggle1.out',
              'logs/example_logs/figure7_8_kaggle2.out',
              'logs/example_logs/figure7_8_kaggle3.out'
             ]

    for fname in files:
        with open(fname, 'r') as f:
            exp_lines = f.readlines()
            for line in exp_lines:
                lines.append(line)

    breaker_ids = []
    for i, line in enumerate(lines):
        if breaker in line :
            breaker_ids.append(i)

    exp_template = { 'tail_time'    : None,
                     'ndcg64'       : None,
                     'model'        : None,
                     'stages'       : None,
                     'batch_size'   : None,
                     'arrival_rate' : None,
                     'gpu_flags'    : None,
                   }

    exps = []
    start_poses = breaker_ids[::2]
    end_poses = breaker_ids[1::2]

    for start_pos, end_pos in zip(start_poses, end_poses):

        exp_lines = range(start_pos, end_pos)
        exp = deepcopy(exp_template)

        for idx in exp_lines:
            line = lines[idx]

            if 'batch_size' in line:
                line = line.rstrip().split(' ')
                num_batches = len(line) - 2
                batch_sizes = []
                for i in range(num_batches):
                    if line[2+i][0] == "[":
                        batch_size = int(line[2+i][1:-2])
                    else:
                        batch_size = int(line[2+i][0:-2])
                    batch_sizes.append(round(batch_size, 5))
                exp['batch_size'] = batch_sizes

            elif 'DCGs @ 64:' in line:
                line = line.rstrip().split(' ')
                dcg = float(line[-1])
                exp['ndcg64'] = round(dcg, 5)

            elif 'model config' in line:
                line = line.rstrip().split(' ')
                model = str(line[-1])
                exp['model'] = model[:-1]
                stages = len(model.split(','))
                exp['stages'] = stages

            elif 'Arrival Rate' in line:
                arr = float(line.rstrip().split()[-1][:-1])
                exp['arrival_rate'] = arr

            elif 'GPU Flags' in line:
                flags = line.rstrip().split(" ")[2:]
                gpu_flags = []
                for flag in flags:
                    flag = flag.replace("[", "")
                    flag = flag.replace("]", "")
                    flag = flag.replace(",", "")
                    if "True" == flag:
                        flag = True
                    elif "False" == flag:
                        flag = False
                    else:
                        print("Incorrect GPU Flag")
                        sys.exit()
                    gpu_flags.append(flag)
                exp['gpu_flags'] = gpu_flags

            elif 'Total time' in line:
                line = line.rstrip().split(' ')
                mean_time = float(line[-2][:-1])
                tail_time = float(line[-1][:-1])
                exp['tail_time'] = round(tail_time, 5)

        exps.append(exp)

    models      = set(list(map(lambda x: x['model'], exps)))
    batch_sizes = set(list(map(lambda x: tuple(x['batch_size']), exps)))
    gpu_flags   = set(list(map(lambda x: tuple(x['gpu_flags']), exps)))
    arrs        = sorted(list(set(list(map(lambda x: x['arrival_rate'], exps)))))

    cpu_1stage  = list(filter(lambda x: x['gpu_flags'] == [False], exps))
    cpu_2stage  = list(filter(lambda x: x['gpu_flags'] == [False, False], exps))
    cpu_3stage  = list(filter(lambda x: x['gpu_flags'] == [False, False, False], exps))

    gpu_1stage  = list(filter(lambda x: True in x['gpu_flags'] and len(x['gpu_flags'])==1, exps))
    gpu_2stage  = list(filter(lambda x: True in x['gpu_flags'] and len(x['gpu_flags'])==2, exps))
    gpu_3stage  = list(filter(lambda x: True in x['gpu_flags'] and len(x['gpu_flags'])==3, exps))

    print("-------------------------------")
    print("Results: Iso-quality (Figure 7-top)")
    print("-------------------------------")

    print("-------")
    print('CPU Two stage results:')
    print("-------")
    cpu_2stage = list(filter(lambda x: x['ndcg64'] > 0.9225, cpu_2stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_2stage))
    arrs = list(map(lambda x: x['arrival_rate'], cpu_2stage))
    _, _, pareto_cpu2_stage = pareto_frontier_z(tail_latencies, arrs, cpu_2stage, maxX=False, maxY=False)
    for exp in pareto_cpu2_stage:
        model = exp['model'].split(",")
        fm = model_to_rm(model[0])
        bm = model_to_rm(model[1])
        s = 1000. / exp['arrival_rate']
        t = exp['tail_time']
        print("Dashed red line Model: {}/{}, Throughput:{}, Tail-latency:{} ms".format(fm, bm, s, t))

    print("-------")
    print('GPU Two stage results:')
    print("-------")
    gpu_2stage = list(filter(lambda x: x['ndcg64'] > 0.9225, gpu_2stage))
    tail_latencies = list(map(lambda x: x['tail_time'], gpu_2stage))
    arrs = list(map(lambda x: x['arrival_rate'], gpu_2stage))
    _, _, pareto_gpu2_stage = pareto_frontier_z(tail_latencies, arrs, gpu_2stage, maxX=False, maxY=False)
    for exp in pareto_gpu2_stage:
        model = exp['model'].split(",")
        fm = model_to_rm(model[0])
        bm = model_to_rm(model[1])
        s = 1000. / exp['arrival_rate']
        t = exp['tail_time']
        print("Solid red line Model: {}/{}, Throughput:{}, Tail-latency:{} ms".format(fm, bm, s, t))

    print("-------")
    print('GPU One stage results:')
    print("-------")
    gpu_1stage = list(filter(lambda x: x['ndcg64'] > 0.9225, gpu_1stage))
    tail_latencies = list(map(lambda x: x['tail_time'], gpu_1stage))
    arrs = list(map(lambda x: x['arrival_rate'], gpu_1stage))
    _, _, pareto_gpu1_stage = pareto_frontier_z(tail_latencies, arrs, gpu_1stage, maxX=False, maxY=False)
    for exp in pareto_gpu1_stage:
        model = exp['model'].split(",")
        m = model_to_rm(model[0])
        s = 1000. / exp['arrival_rate']
        t = exp['tail_time']
        print("Black line Model: {}, Throughput:{}, Tail-latency:{} ms".format(m, s, t))



    print("-------------------------------")
    print("Results: Iso-throughput (Figure 7-bottom)")
    print("-------------------------------")
    print("-------")
    print('CPU Two stage results:')
    print("-------")
    cpu_2stage  = list(filter(lambda x: x['gpu_flags'] == [False, False], exps))
    cpu_2stage = list(filter(lambda x: x['arrival_rate'] <= 15, cpu_2stage))
    ndcgs = list(map(lambda x: x['ndcg64'], cpu_2stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_2stage))
    _, _, pareto_cpu2_stage = pareto_frontier_z(tail_latencies, ndcgs, cpu_2stage, maxX=False, )
    for exp in pareto_cpu2_stage:
        model = exp['model'].split(",")
        m = model_to_rm(model[0])
        n = exp['ndcg64'] * 100
        t = exp['tail_time']
        print("Red line Model: {}, NDCG:{} %, Tail-latency:{} ms".format(m, n, t))

    print("-------")
    print('GPU One stage results:')
    print("-------")
    gpu_2stage  = list(filter(lambda x: x['gpu_flags'] == [True], exps))
    gpu_2stage = list(filter(lambda x: x['arrival_rate'] <= 15, gpu_2stage))
    ndcgs = list(map(lambda x: x['ndcg64'], gpu_2stage))
    tail_latencies = list(map(lambda x: x['tail_time'], gpu_2stage))
    _, _, pareto_gpu2_stage = pareto_frontier_z(tail_latencies, ndcgs, gpu_2stage, maxX=False, )
    for exp in pareto_gpu2_stage:
        model = exp['model'].split(",")
        m = model_to_rm(model[0])
        n = exp['ndcg64'] * 100
        t = exp['tail_time']
        print("Black line Model: {}, NDCG:{} %, Tail-latency:{} ms".format(m, n, t))

if __name__=="__main__":
    main()
