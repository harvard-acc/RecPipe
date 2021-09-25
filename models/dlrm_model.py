# numpy
import numpy as np
import sys

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

from utils import time_wrap, load_model

import time
import math
import os

# data generation
import dlrm_data_pytorch as dp

# Recommendation model implementation for Criteo Kaggle dataset for RecPipe
# built off of Facebook's Deep Learning Recommendation Model (DLRM):
# https://github.com/facebookresearch/dlrm
class DLRM_Net(nn.Module):

    def dlrm_wrap(self, X, lS_o, lS_i, use_gpu, device):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(device)
            return self(
                X.to(device),
                lS_o,
                lS_i
            )
        else:
            return self(X, lS_o, lS_i)

    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(n, m, self.qr_collisions,
                    operation=self.qr_operation, mode="sum", sparse=True)
            elif self.md_flag:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)

            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)

            emb_l.append(EE)

        return emb_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold
            # create operators
            if ndevices <= 1:
                self.emb_l = self.create_emb(m_spa, ln_emb)
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)
            self.sample_id = 0

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        #ly = [ torch.tensor(np.zeros( (4096, 4), dtype=np.float32)) for _ in range(len(lS_i)) ]
        ly  = []

        for k, sparse_index_group_batch in enumerate(lS_i):
            #if k > 0:
            #    continue
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            #print("V : ", V)
            #print("o : ", lS_o[k]),
            #print("i : ", lS_i[k])
            #sys.exit()
            #ly[k] = V

            ly.append(V)

        #print(np.array(ly).shape)
        #print(ly[0])
        #ly[0] = torch.tensor( np.zeros( (4096, 4) , dtype=np.float32) )
        #print(ly[0])
        #print(ly[1])
        #sys.exit()
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        return self.sequential_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        #print("Running top MLP layer", z.size())
        #print("Top MLP layer", self.top_l)


        p = self.apply_mlp(z, self.top_l)

        #root_dir = "../RP_Accel_samples/sample_" + str(self.sample_id)
        #os.mkdir(root_dir)
        #np.save(root_dir + "/input", np.array(z.data))

        #for idx, param in enumerate(self.top_l.parameters()):
        #    print(param.name, param.data, param.size())
        #    np.save(root_dir + "/weights_" + str(idx), np.array(param.data))

        #np.save(root_dir + "/output", np.array(p.data))

        # clamp output if needed
        self.sample_id += 1

        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z


    def test(self, args, test_ld, device):
        use_gpu = args.use_gpu and torch.cuda.is_available()
        # don't measure training iter time in a test iteration
        if args.mlperf_logging:
            previous_iteration_time = None

        test_accu = 0
        test_loss = 0
        test_samp = 0
        test_iter = 0
        best_gA_test = 0
        best_auc_test = 0
        total_time = 0
        nbatches_test = len(test_ld)

        accum_test_time_begin = time_wrap(use_gpu)
        if args.mlperf_logging:
            scores = []
            targets = []

        for j, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
            t1_test = time_wrap(use_gpu)

            # forward pass
            Z_test = self.dlrm_wrap(
                X_test, lS_o_test, lS_i_test, use_gpu, device
            )

            # loss
            E_test = loss_fn_wrap(args, Z_test, T_test, use_gpu, device)

            # compute loss and accuracy
            L_test = E_test.detach().cpu().numpy()  # numpy array
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

            t2_test = time_wrap(use_gpu)
            total_time += t2_test - t1_test

            test_accu += A_test
            test_loss += L_test * mbs_test
            test_samp += mbs_test
            test_iter += 1

            gA_test = test_accu / test_samp
            gL_test = test_loss / test_samp

            # print time, loss and accuracy
            should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches_test)
            if should_print:
                gA = test_accu / test_samp
                test_accu = 0

                gL = test_loss / test_samp
                test_loss = 0

                gT = 1000.0 * total_time / test_iter if args.print_time else -1
                total_time = 0

                str_run_type = "testing"
                print(
                    "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                        str_run_type, j + 1, nbatches_test, 0, gT
                    )
                    + "loss {:.6f}, accuracy {:3.3f} %".format(gL, gA * 100)
                )

                # Uncomment the line below to print out the total time with overhead
                # print("Accumulated time so far: {}" \
                # .format(time_wrap(use_gpu) - accum_time_begin))
                test_iter = 0
                test_samp = 0


        return


    def test_inline(self, args, X_int_test, X_cat_test, num_samples = None):
        use_gpu = args.use_gpu and torch.cuda.is_available()

        if use_gpu:
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")

        num_cat_features = 26
        num_int_features = 13

        t_X_lens = [torch.tensor(range(args.test_mini_batch_size)) for _ in range(num_cat_features)]

        if num_samples:
            samples = num_samples
        else:
            samples = len(X_cat_test)
        max_iterations = int(math.floor(samples / args.test_mini_batch_size))

        Z_aggregate = np.array([], dtype=np.float16)

        preprocess_time = time.time()

        data_times      = 0
        inference_times = 0
        fetch_times     = 0

        for i in range(max_iterations):
            iter_start_time = time.time()

            start_idx = i * args.test_mini_batch_size
            end_idx = (i+1)  * args.test_mini_batch_size
            np_X_int = X_int_test[start_idx : end_idx]
            t_X_int = torch.log(torch.tensor(np_X_int, dtype=torch.float)+1)

            np_lS_i = X_cat_test[start_idx: end_idx, :]
            t_X_cat = torch.tensor(np_lS_i, dtype=torch.long)

            iter_data_time = time.time()

            z = self.dlrm_wrap(t_X_int, t_X_lens, t_X_cat.T, use_gpu, device)

            iter_inference_time = time.time()

            if args.use_gpu:
                Z_aggregate = np.concatenate( (Z_aggregate, z.cpu().detach().numpy()), axis=None)
            else:
                Z_aggregate = np.concatenate( (Z_aggregate, z.detach().numpy()), axis=None)

            iter_fetch_time = time.time()

            data_times += (iter_data_time - iter_start_time)
            inference_times += (iter_inference_time - iter_data_time)
            fetch_times += (iter_fetch_time - iter_inference_time)

        return np.array(Z_aggregate, dtype=np.float16), preprocess_time, data_times, inference_times, fetch_times



def construct_dlrm_model(args, data_set_only):
    if data_set_only:
        use_gpu = False
    else:
        use_gpu = args.use_gpu and torch.cuda.is_available()

    if use_gpu and (not data_set_only):
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")
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

    if not data_set_only:
        ### construct the neural network specified above ###
        # WARNING: to obtain exactly the same initialization for
        # the weights we need to start from the same random seed.
        # np.random.seed(args.numpy_rand_seed)
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=ln_top.size - 2,
            sync_dense_params=args.sync_dense_params,
            loss_threshold=args.loss_threshold,
            ndevices=ndevices,
        )

        if use_gpu:
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            dlrm = dlrm.to(device)  # .cuda()
            if dlrm.ndevices > 1:
                dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)

        # Load model is specified
        if not (args.load_model == ""):
            load_model(args, dlrm)

    if not data_set_only:
        return dlrm, test_data, test_ld
    else:
        return None, test_data, test_ld

    return
