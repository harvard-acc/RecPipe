from copy import deepcopy

def model_to_rm(string):
    if 'dim_4_0' in string:
        return 'RM-small'
    if 'dim_16' in string:
        return 'RM-med'
    if 'dim_32' in string:
        return 'RM-large'


def main():
    breaker = "================================================"

    lines  = []
    files = ['logs/example_logs/figure3.out']

    for fname in files:
        with open(fname, 'r') as f:
            exp_lines = f.readlines()
            for line in exp_lines:
                lines.append(line)

    breaker_ids = []
    for i, line in enumerate(lines):
        if breaker in line :
            breaker_ids.append(i)

    exp_template = { 'mean_time': None,
                     'tail_time': None,
                     'ndcg4': None,
                     'ndcg8': None,
                     'ndcg16': None,
                     'ndcg32': None,
                     'ndcg64': None,
                     'error': None,
                     'model': None,
                     'batch_size': None,
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
                        batch_size = int(line[2+i][0:-1])
                    batch_sizes.append(round(batch_size, 5))
                exp['batch_size'] = batch_sizes
            elif 'DCGs @ 64:' in line:
                line = line.rstrip().split(' ')
                dcg = float(line[-1])
                exp['ndcg64'] = round(dcg, 5)
            elif 'model config' in line:
                line = line.rstrip().split(' ')
                model = str(line[-1])
                exp['model'] = model
            elif 'Error' in line:
                line = line.rstrip().split(' ')
                error = float(line[-1])
                exp['error'] = error

        exps.append(exp)

    print("-------------------------------")
    print("Results: accuracy tradeoffs")
    print("-------------------------------")
    # Print accuracy tradeoffs
    batch_size = 4096.0
    models = ['dim_4_0.json', 'dim_16.json', 'dim_32_0.json']
    label = ['Red bar', 'Blue bar', 'Black bar']
    for i, model in enumerate(models):
        for exp in exps:
            if model in exp['model'] and batch_size == exp['batch_size'][0]:
                print("{} Model: {}, Error: {} %".format(label[i],
                                                         model_to_rm(exp['model']),
                                                         exp['error']*100.))

    print("-------------------------------")
    print("Results: RM-small NDCG tradeoffs")
    print("-------------------------------")
    # Print ndcg tradeoffs
    batch_size = [256.0, 512.0, 1024.0, 2048., 4096.0]
    model = 'dim_4_0.json'
    for batch in batch_size:
        for exp in exps:
            if model in exp['model'] and batch == exp['batch_size'][0]:
                print("RM-Small batch: {}, NDCG: {} %".format(batch, exp['ndcg64']*100.))

    print("-------------------------------")
    print("Results: RM-small vs. med vs. large")
    print("-------------------------------")
    # Print ndcg tradeoffs
    batch_size = 4096.0
    models = ['dim_4_0.json', 'dim_16.json', 'dim_32_0.json']
    label = ['Red bar', 'Blue bar', 'Black bar']
    for i, model in enumerate(models):
        for exp in exps:
            if model in exp['model'] and batch_size == exp['batch_size'][0]:
                print("{} Model: {}, NDCG: {} %".format(label[i],
                                                        model_to_rm(exp['model']),
                                                        exp['ndcg64']*100.))

if __name__=="__main__":
    main()
