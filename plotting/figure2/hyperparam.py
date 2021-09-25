
from mpl_plot import *
import json
import glob


def mlp_to_flops(arch_mlp):
    flops = 0

    arch_mlp = arch_mlp.rstrip().split('-')
    for i in range(len(arch_mlp)-1):
        flops += (int(arch_mlp[i]) * int(arch_mlp[i+1]))

    return flops


def pareto_frontier(Xs, Ys, Zs, maxX = True, maxY=True):
    myList = sorted([ [Xs[i], Ys[i], Zs[i]] for i in range(len(Xs))], reverse=maxX)

    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY:
            if pair[1] > p_front[-1][1]:
                if pair[0] == p_front[-1][0]:
                    p_front[-1] = pair
                else:
                    p_front.append(pair)
        else:
            if pair[1] < p_front[-1][1]:
                if pair[0] == p_front[-1][0]:
                    p_front[-1] = pair
                else:
                    p_front.append(pair)

    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    p_frontZ = [pair[2] for pair in p_front]

    return p_frontX, p_frontY, p_frontZ

def get_floats(datas):
    return list(map(lambda x: x[0], datas))

def get_accs(datas):
    return list(map(lambda x: x[1], datas))


def get_err(datas):
    return list(map(lambda x: 100. - x[1], datas))

def main():
    data_dir = "../model_outputs/"

    files = glob.glob(data_dir + "/*")

    exp_flops  = []
    exp_accs   = []
    exp_sparse = []
    fpaths = []

    for fpath in files:
        with open(fpath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 1:
                continue
            if 'Testing' not in lines[-1]:
                continue

            cli = lines[0][20:].rstrip()
            cli = json.loads(cli)

            mlp_top = cli['arch_mlp_top']
            mlp_bot = cli['arch_mlp_bot']
            sparse_feat_dim = cli['arch_sparse_feature_size']

            mlp_top_flops = mlp_to_flops(mlp_top)
            mlp_bot_flops = mlp_to_flops(mlp_bot)
            flops = mlp_top_flops + mlp_bot_flops

            acc = float(lines[-1].rstrip().split(' ')[-2])
            fpaths.append(fpath)

            exp_flops.append(flops)
            exp_accs.append(acc)
            exp_sparse.append(sparse_feat_dim)

    #dim_32 params
    #exp_flops.append(270*1000)
    #exp_accs.append(78.84)
    #exp_sparse.append(32)

    p_flops, p_acc, p_sparse = pareto_frontier(exp_flops, exp_accs, exp_sparse,
                                               maxX = False, maxY = True)

    p_flops, p_acc, fpaths = pareto_frontier(exp_flops, exp_accs, fpaths,
                                               maxX = False, maxY = True)

    print("Pareto Optimal Model Configurations")
    print("FLOPs, Accuracy (%), Embedding vector, Filepath")
    for f, a, s, n in zip(p_flops, p_acc, p_sparse, fpaths):
        print(f, a, s, n)

    fig, ax1 = plt.subplots(figsize = (8,4.5))
    #luminance channel sweeps from dark to light, (for ordered comparisons)
    colors = [teal, 'k', orange, blue, crimson]

    sparse_2  = list(filter(lambda x: x[2] == 2, zip(exp_flops, exp_accs, exp_sparse)))
    sparse_4  = list(filter(lambda x: x[2] == 4, zip(exp_flops, exp_accs, exp_sparse)))
    sparse_8  = list(filter(lambda x: x[2] == 8, zip(exp_flops, exp_accs, exp_sparse)))
    sparse_16 = list(filter(lambda x: x[2] == 16, zip(exp_flops, exp_accs, exp_sparse)))
    sparse_32 = list(filter(lambda x: x[2] == 32, zip(exp_flops, exp_accs, exp_sparse)))

    plt.scatter(get_floats(sparse_2), get_err(sparse_2), color=colors[0]  , alpha=1.)
    plt.scatter(get_floats(sparse_4), get_err(sparse_4), color=colors[1]  , alpha=1.)
    plt.scatter(get_floats(sparse_8), get_err(sparse_8), color=colors[2]  , alpha=1.)
    plt.scatter(get_floats(sparse_16), get_err(sparse_16), color=colors[3], alpha=1.)
    plt.scatter(get_floats(sparse_32), get_err(sparse_32), color=colors[4], alpha=1.)

    p_err = list(map(lambda x: 100. - x, p_acc))
    plt.plot(p_flops, p_err, linestyle='--', linewidth=1.0, color='k')

    plt.xlabel('FLOPs (log)', )
    plt.ylabel('Test error (%)', )

    fig.tight_layout()
    filename = 'hyperparm'

    plt.ylim([100. - 78.9, 100. - 78.5])
    plt.xlim([750, 300*1000])
    plt.xscale('log')
    plt.savefig(filename + '.pdf')
    plt.close()
    print("Plot saved in {}.pdf".format(filename))


if __name__=="__main__":
    main()
