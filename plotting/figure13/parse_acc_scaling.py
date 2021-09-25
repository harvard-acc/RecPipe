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

def parse_cpu_gpus(files):
    breaker = "================================"
    lines  = []

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
                    batch_size = int(clean_string(line[2+i]))
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

            elif 'Total time' in line:
                line = line.rstrip().split(' ')
                tail_time = float(clean_string(line[-1]))
                exp['tail_time'] = round(tail_time, 5)

        exps.append(exp)
    return exps

def parse_accels(files):
    breaker = "================================"
    lines  = []

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

        exp['ndcg64']=92.4
        exps.append(exp)

    return exps


def main():

    cpugpu_files  = ['logs/example_logs/figure13_acc.out', ]

    accel_files  = ['logs/example_logs/figure13_accel_1_stage.out',
                    'logs/example_logs/figure13_accel_2_stage.out',
                  ]

    exps       = parse_cpu_gpus(cpugpu_files)
    accel_exps = parse_accels(accel_files)

    cpu_1stage  = list(filter(lambda x: x['gpu_flags'] == [False], exps))

    accel_1stage  = list(filter(lambda x: x['stages'] == 1, accel_exps))
    accel_2stage  = list(filter(lambda x: x['stages'] == 2, accel_exps))

    print("-------------------------------")
    print("Results: Overall results for Criteo Kaggle data set (Figure 14)")
    print("-------------------------------")
    # Lower arrival rate (higher throughput) degrades tail-latency of RM-large
    # more than other configurations

    experiments = [ accel_1stage, accel_2stage, ]
    labels = [ "Accel 1 stage", "Accel 2 stage", ]

    batch_to_acc = { }

    for exp in cpu_1stage:
        batch_to_acc[exp['batch_size'][0]] = exp['ndcg64']

    arrival_rates = [4.0]
    aids = ["Low", "Medium", "High"]
    ndcg_thres = 92.25

    for aid, arrival_rate in enumerate(arrival_rates):
        for i in range(len(experiments)):
            for batch in sorted(batch_to_acc.keys()):

                exp = list(filter(lambda x: x['arrival_rate'] == arrival_rate, experiments[i]))
                exp = list(filter(lambda x: batch in x['batch_size'] , exp))
                exp = sorted(exp, key=lambda x: x['tail_time'])
                if len(exp) == 0:
                    continue
                if exp[0]['tail_time'] * 1000. > 500:
                    print(labels[i], "Unachievable")
                else:
                    print(labels[i], exp[0]['accel_configs'], exp[0]['batch_size'], exp[0]['tail_time'] * 1000. , "ms")
            print("--------")

if __name__=="__main__":
    main()
