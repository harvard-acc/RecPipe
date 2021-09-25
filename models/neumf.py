import numpy as np
import torch
import torch.nn as nn
import time

#from mlperf_compliance import mlperf_log

class NeuMF(nn.Module):
    def __init__(self, nb_users, nb_items,
                 mf_dim, mf_reg,
                 mlp_layer_sizes, mlp_layer_regs):
        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError('u dummy, layer_sizes != layer_regs!')
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NeuMF, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)

        #mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_MF_DIM, value=mf_dim)

        # TODO: regularization?
        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)

        #mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_MLP_LAYER_SIZES, value=mlp_layer_sizes)
        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])  # noqa: E501

        self.final = nn.Linear(mlp_layer_sizes[-1] + mf_dim, 1)

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)
        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)
        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)

        x = torch.cat((xmf, xmlp), dim=1)
        x = self.final(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x

def construct_movielens_model(args, data_set_only=False):
    if data_set_only:
        use_gpu = False
    else:
        use_gpu = args.use_gpu and torch.cuda.is_available()

    if use_gpu and (not data_set_only):
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    if not data_set_only:
        model = torch.load(args.load_model, map_location=torch.device(device))

        # Generate MovieLens dataset paths from users/item data
        dataset_path = args.processed_data_file
        user_data = dataset_path + "/test_users.t7"
        item_data = dataset_path + "/test_items.t7"

        # Generate MovieLens dataset paths from users/item data
        test_users = torch.load(user_data, map_location=torch.device('cpu'))
        test_items = torch.load(item_data, map_location=torch.device('cpu'))

        test_data = (test_users, test_items)

        return model, test_data, None
    else:
        # Generate MovieLens dataset paths from users/item data
        dataset_path = args.processed_data_file
        user_data = dataset_path + "/test_users.t7"
        item_data = dataset_path + "/test_items.t7"

        # Generate MovieLens dataset paths from users/item data
        test_users = torch.load(user_data, map_location=torch.device('cpu'))
        test_items = torch.load(item_data, map_location=torch.device('cpu'))

        test_data = (test_users, test_items)

        return None, test_data, None


def wrap_movielens_inference( args, model, users, items, batch_size  ):
    use_gpu = args.use_gpu and torch.cuda.is_available()

    if use_gpu:
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    preprocess_time = time.time()

    if use_gpu:
        res = model(users.to(device), items.to(device), sigmoid=True).detach().view(-1,batch_size)
        e = time.time()

        res = res.cpu().detach().numpy()

        f = time.time()
        inference_time = e - preprocess_time
        fetch_time = f - e
    else:
        res = model(users, items, sigmoid=True).detach().view(-1,batch_size)
        e = time.time()
        inference_time = e - preprocess_time
        fetch_time = 0

    return np.array(res[0], dtype=np.float16), preprocess_time, 0, inference_time, fetch_time
