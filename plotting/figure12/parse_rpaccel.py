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

def clean_string(s):
    s = s.replace('[', '')
    s = s.replace(',', '')
    s = s.replace(' ', '')
    s = s.replace(']', '')
    s = s.replace(')', '')
    return s

def main():
    breaker = "================================"

    lines  = []
    #################################################################
    # TODO: Modify these lines to point to experimentally generated output
    # files to generate results for new data. Currently this points to example
    # logs provided in the artifact.
    #################################################################
    files  = [
              '../../logs/example_logs/figure12_kaggle1.out',
              '../../logs/example_logs/figure12_kaggle2.out',
              '../../logs/example_logs/figure12_kaggle3.out'
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
                     'accel_config' : None,
                     'accel_counts' : None,
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
                    batch_size = int(clean_string(line[2+i]))
                    batch_sizes.append(round(batch_size, 5))
                exp['batch_size'] = batch_sizes

            elif 'accel counts' in line:
                line = line.rstrip().split(' ')
                num_accels = len(line) - 3
                accel_counts = []
                for i in range(num_accels):
                    accel_count = int(clean_string(line[3+i]))
                    accel_counts.append(round(accel_count, 5))
                exp['accel_counts'] = accel_counts

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

            elif 'Arrival rate' in line:
                arr = float(line.rstrip().split()[-1][:-1])
                exp['arrival_rate'] = arr

            elif 'GPU Flags' in line:
                flags = line.rstrip().split(" ")[2:]
                gpu_flags = []
                for flag in flags:
                    flag = clean_string(flag)
                    if "True" == flag:
                        flag = True
                    elif "False" == flag:
                        flag = False
                    else:
                        print("Incorrect GPU Flag")
                        sys.exit()
                    gpu_flags.append(flag)
                exp['gpu_flags'] = gpu_flags

            elif 'accel config' in line:
                configs = line.rstrip().split(" ")[2:]
                accel_configs = []
                for config in configs:
                    config = clean_string(config)
                    accel_configs.append(config)
                exp['accel_configs'] = accel_configs

            elif 'Total tail time' in line:
                line = line.rstrip().split(' ')
                tail_time = float(clean_string(line[-1]))
                exp['tail_time'] = round(tail_time, 5)

        if exp['arrival_rate'] == None:
            continue

        if exp['tail_time'] == None:
            continue

        exps.append(exp)

    models      = set(list(map(lambda x: x['model'], exps)))
    arrs        = sorted(list(set(list(map(lambda x: x['arrival_rate'], exps)))))

    accel_1stage  = list(filter(lambda x: x['stages'] == 1, exps))
    accel_2stage  = list(filter(lambda x: x['stages'] == 2, exps))
    accel_3stage  = list(filter(lambda x: x['stages'] == 3, exps))

    print("-------------------------------")
    print("Results: Multi-stage RPAccel (Figure 12-top)")
    print("-------------------------------")

    print("-------")
    print('RPAccel One stage results (black line):')
    print("-------")
    tail_latencies = list(map(lambda x: x['tail_time'], accel_1stage))
    arrs = list(map(lambda x: x['arrival_rate'], accel_1stage))
    pareto_tail, pareto_arrs, _ = pareto_frontier_z(tail_latencies, arrs,
                                                    accel_1stage,
                                                    maxX=False, maxY=False)

    for t, arr in zip(pareto_tail, pareto_arrs):
        s = 1000. / arr
        t = t * 1000.
        if t > 100:
            print("Throughput:{}, Unachievable".format(s, t))
        else:
            print("Throughput:{}, Tail-latency:{} ms".format(s, t))


    print("-------")
    print('RPAccel Two stage results (red line):')
    print("-------")
    tail_latencies = list(map(lambda x: x['tail_time'], accel_2stage))
    arrs = list(map(lambda x: x['arrival_rate'], accel_2stage))
    pareto_tail, pareto_arrs, _= pareto_frontier_z(tail_latencies, arrs, accel_2stage, maxX=False, maxY=False)

    for t, arr in zip(pareto_tail, pareto_arrs):
        s = 1000. / arr
        t = t * 1000.
        if t > 100:
            print("Throughput:{}, Unachievable".format(s, t))
        else:
            print("Throughput:{}, Tail-latency:{} ms".format(s, t))

    print("-------")
    print('RPAccel Three stage results (blue line):')
    print("-------")
    tail_latencies = list(map(lambda x: x['tail_time'], accel_3stage))
    arrs = list(map(lambda x: x['arrival_rate'], accel_3stage))
    pareto_tail, pareto_arrs, _ = pareto_frontier_z(tail_latencies, arrs,
                                                   arrs,
                                                   maxX=False, maxY=False)

    for t, arr in zip(pareto_tail, pareto_arrs):
        s = 1000. / arr
        t = t * 1000.
        if t > 100:
            print("Throughput:{}, Unachievable".format(s, t))
        else:
            print("Throughput:{}, Tail-latency:{} ms".format(s, t))

if __name__=="__main__":
    main()
