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
    #################################################################
    # TODO: Modify these lines to point to experimentally generated output
    # files to generate results for new data. Currently this points to example
    # logs provided in the artifact.
    #################################################################
    files  = ['../../logs/example_logs/figure7_8_kaggle1.out',
              '../../logs/example_logs/figure7_8_kaggle2.out',
              '../../logs/example_logs/figure7_8_kaggle3.out'
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

    print("-------------------------------")
    print("Results: CPU Single Stage (Figure 6a)")
    print("-------------------------------")
    # Lower arrival rate (higher throughput) degrades tail-latency of RM-large
    # more than other configurations
    figure6_parta = list(filter(lambda x: x['arrival_rate'] == 1.0, cpu_1stage))
    models = ['dim_4_0', 'dim_16', 'dim_32_0']
    label = ['Light grey line', 'Medium grey line', 'Black line']
    for i, model in enumerate(models):
        for exp in figure6_parta:
            if (model in exp['model']) and (exp['stages'] == 1):
                m = model_to_rm(exp['model'])
                b = int(exp['batch_size'][0])
                n = exp['ndcg64'] * 100.
                t = exp['tail_time']
                print("{} Model: {}, Batch:{}, NDCG:{} %, Tail-latency:{} ms".format(label[i],
                                                                                     m, b, n, t))
        print

    print("-------------------------------")
    print("Results: CPU Stages (Figure 6b)")
    print("-------------------------------")
    print('One stage results:')
    print("-------")
    cpu_1stage = list(filter(lambda x: x['arrival_rate'] == 1.0, cpu_1stage))
    ndcgs = list(map(lambda x: x['ndcg64'], cpu_1stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_1stage))
    _, _, pareto_cpu1_stage = pareto_frontier_z(tail_latencies, ndcgs, cpu_1stage, maxX=False)
    for exp in pareto_cpu1_stage:
        model = exp['model'].split(",")
        m = model_to_rm(model[0])
        n = exp['ndcg64'] * 100.
        t = exp['tail_time']
        print("Black line Model: {}, NDCG:{} %, Tail-latency:{} ms".format(m, n, t))

    print("-------")
    print('Two stage results:')
    print("-------")
    cpu_2stage = list(filter(lambda x: x['arrival_rate'] == 1.0, cpu_2stage))
    ndcgs = list(map(lambda x: x['ndcg64'], cpu_2stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_2stage))
    _, _, pareto_cpu2_stage = pareto_frontier_z(tail_latencies, ndcgs, cpu_2stage, maxX=False)
    for exp in pareto_cpu2_stage:
        model = exp['model'].split(",")
        fm = model_to_rm(model[0])
        bm = model_to_rm(model[1])
        n = exp['ndcg64'] * 100.
        t = exp['tail_time']
        print("Red line Model: {}/{}, NDCG:{} %, Tail-latency:{} ms".format(fm, bm, n, t))

    print("-------")
    print('Three stage results:')
    print("-------")
    cpu_3stage = list(filter(lambda x: x['arrival_rate'] == 1.0, cpu_3stage))
    ndcgs = list(map(lambda x: x['ndcg64'], cpu_3stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_3stage))
    _, _, pareto_cpu3_stage = pareto_frontier_z(tail_latencies, ndcgs, cpu_3stage, maxX=False)
    for exp in pareto_cpu3_stage:
        model = exp['model'].split(",")
        fm = model_to_rm(model[0])
        mm = model_to_rm(model[1])
        bm = model_to_rm(model[2])
        n = exp['ndcg64'] * 100.
        t = exp['tail_time']
        print("Blue line Model: {}/{}/{}, NDCG:{} %, Tail-latency:{} ms".format(fm, mm, bm, n, t))

    print("-------------------------------")
    print("Results: CPU Stages (Figure 6c)")
    print("-------------------------------")
    print('One stage results:')
    print("-------")
    cpu_1stage  = list(filter(lambda x: x['gpu_flags'] == [False], exps))
    cpu_1stage = list(filter(lambda x: x['ndcg64'] > 0.9225, cpu_1stage))
    arrs = list(map(lambda x: x['arrival_rate'], cpu_1stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_1stage))
    _, _, pareto_cpu1_stage = pareto_frontier_z(arrs, tail_latencies, cpu_1stage, maxX=False, maxY=False)
    for exp in pareto_cpu1_stage:
        model = exp['model'].split(",")
        m = model_to_rm(model[0])
        s = 1000. / exp['arrival_rate']
        t = exp['tail_time']
        print("Black line Model: {}, Throughout:{} %, Tail-latency:{} ms".format(m, s, t))

    print('Two stage results:')
    print("-------")
    cpu_2stage  = list(filter(lambda x: x['gpu_flags'] == [False, False], exps))
    cpu_2stage = list(filter(lambda x: x['ndcg64'] > 0.9225, cpu_2stage))
    arrs = list(map(lambda x: x['arrival_rate'], cpu_2stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_2stage))
    _, _, pareto_cpu2_stage = pareto_frontier_z(arrs, tail_latencies, cpu_2stage, maxX=False, maxY=False)
    for exp in pareto_cpu2_stage:
        model = exp['model'].split(",")
        fm = model_to_rm(model[0])
        bm = model_to_rm(model[1])
        s = 1000. / exp['arrival_rate']
        t = exp['tail_time']
        print("Red line Model: {}/{}, Throughout:{} %, Tail-latency:{} ms".format(fm, bm, s, t))

    print('Three stage results:')
    print("-------")
    cpu_3stage  = list(filter(lambda x: x['gpu_flags'] == [False, False, False], exps))
    cpu_3stage = list(filter(lambda x: x['ndcg64'] > 0.9225, cpu_3stage))
    arrs = list(map(lambda x: x['arrival_rate'], cpu_3stage))
    tail_latencies = list(map(lambda x: x['tail_time'], cpu_3stage))
    _, _, pareto_cpu3_stage = pareto_frontier_z(arrs, tail_latencies, cpu_3stage, maxX=False, maxY=False)
    for exp in pareto_cpu3_stage:
        model = exp['model'].split(",")
        fm = model_to_rm(model[0])
        mm = model_to_rm(model[1])
        bm = model_to_rm(model[2])
        s = 1000. / exp['arrival_rate']
        t = exp['tail_time']
        print("Blue line Model: {}/{}/{}, Throughout:{} %, Tail-latency:{} ms".format(fm, mm, bm, s, t))

if __name__=="__main__":
    main()
