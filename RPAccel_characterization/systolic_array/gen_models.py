import os, sys
import json

def gen_model(batch_size, emb_dim, num_emb_tbls, arch_mlp_bot, arch_mlp_top, filename):
    arch_mlp_top = str((num_emb_tbls+1)*(emb_dim))+"-"+arch_mlp_top
    
    arch_mlp_top = arch_mlp_top.rstrip().split('-')
    arch_mlp_bot = arch_mlp_bot.rstrip().split('-')

    file_topology = open(filename, 'w')
    file_topology.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,\n")

    for i in range(len(arch_mlp_bot)-1):
        name          = "MLP_Bot_"+str(i)
        input_height  = str(batch_size)
        input_width   = str(arch_mlp_bot[i])
        filter_height = str(1)
        filter_width  = str(arch_mlp_bot[i])
        channels      = str(1)
        filters       = str(arch_mlp_bot[i+1])
        strides       = str(1)

        file_topology.write(name+","+input_height+","+input_width+","+filter_height+","+filter_width+","
                            +channels+","+filters+","+strides+",\n")

    for i in range(len(arch_mlp_top)-1):
        name          = "MLP_Top_"+str(i)
        input_height  = str(batch_size)
        input_width   = str(arch_mlp_top[i])
        filter_height = str(1)
        filter_width  = str(arch_mlp_top[i])
        channels      = str(1)
        filters       = str(arch_mlp_top[i+1])
        strides       = str(1)

        file_topology.write(name+","+input_height+","+input_width+","+filter_height+","+filter_width+","
                            +channels+","+filters+","+strides+",\n")

    file_topology.close()

if __name__ == '__main__':

    TOPOLOGIES_DIR = "./RPAccel_characterization/systolic_array/topologies/recommendation/"
    MODELS_DIR     = "./configs/model_configs/"

    if os.path.exists(TOPOLOGIES_DIR):
        os.system("rm -rv "+TOPOLOGIES_DIR)
        print("*** Deleting old recommendation topologies!")
    os.system("mkdir "+TOPOLOGIES_DIR)

    batch_sizes   = [4096]

    model_names       = ['dim_4_0.json', 'dim_16.json', 'dim_32_0.json']
    emb_dims          = []
    emb_tbls          = []
    arch_mlp_bot_list = []
    arch_mlp_top_list = []

    for model_name in model_names:
        model = json.load(open(MODELS_DIR+model_name))
        emb_dims.append(model['arch_sparse_feature_size'])
        emb_tbls.append(26)
        arch_mlp_bot_list.append(model['arch_mlp_bot'])
        arch_mlp_top_list.append(model['arch_mlp_top'])


    for batch_size in batch_sizes:
        for model_num in range(len(emb_dims)):
            filename = TOPOLOGIES_DIR+"dlrm_"+str(model_num)+"_e"+str(emb_dims[model_num])+"_bs"+str(batch_size)+".csv"
            gen_model(batch_size, emb_dims[model_num], emb_tbls[model_num], arch_mlp_bot_list[model_num], arch_mlp_top_list[model_num], filename)
