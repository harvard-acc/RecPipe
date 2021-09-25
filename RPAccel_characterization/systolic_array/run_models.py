import os, sys
import json

def gen_config(topology, acc_num, array_dim, i_sram_size, w_sram_size, o_sram_size, data_placement):
    model = topology[0:-4]
    filename = ACC_DIR+"acc_"+str(acc_num)+"_"+model+"_"+data_placement+".cfg"

    file_config = open(filename, 'w')
    file_config.write("[general]\n")

    run_name = "\""+"acc_"+str(acc_num)+"_"+model+"_"+data_placement+"\""
    file_config.write("run_name="+run_name+"\n\n")

    file_config.write("[architecture_presets]\n")
    file_config.write("ArrayHeight:"+str(array_dim[0])+"\n")
    file_config.write("ArrayWidth:"+str(array_dim[1])+"\n")

    file_config.write("IfmapSramSz:"+str(i_sram_size)+"\n")
    file_config.write("FilterSramSz:"+str(w_sram_size)+"\n")
    file_config.write("OfmapSramSz:"+str(o_sram_size)+"\n")

    file_config.write("IfmapOffset:"+str(0)+"\n")
    file_config.write("FilterOffset:"+str(10000000)+"\n")
    file_config.write("OfmapOffset:"+str(20000000)+"\n")

    file_config.write("Dataflow:"+data_placement+"\n")

    file_config.close()

def run_config(topology, acc_num, data_placement):
    model    = topology[0:-4]
    topology = TOPOLOGIES_DIR+topology
    acc      = ACC_DIR+"acc_"+str(acc_num)+"_"+model+"_"+data_placement+".cfg"
    
    cmd   = "python3 ./RPAccel_characterization/systolic_array/scale.py -arch_config="+acc+" -network="+topology

    os.system(cmd)

if __name__ == '__main__':

    TOPOLOGIES_DIR = "./RPAccel_characterization/systolic_array/topologies/recommendation/"
    ACC_DIR        = "./RPAccel_characterization/systolic_array/configs/multistage_acc/"
    OUTPUT_DIR     = "./RPAccel_characterization/systolic_array/outputs/"

    ACC_CONFIGS    = "./RPAccel_characterization/systolic_array/configs/acc_configs_overview.json"

    if not os.path.exists(TOPOLOGIES_DIR):
        print("*** Topologies do not exist!")

    if os.path.exists(ACC_DIR):
        os.system("rm -rv "+ACC_DIR)
        print("*** Deleting old accelerator configurations")
    os.system("mkdir "+ACC_DIR)
        
    if os.path.exists(OUTPUT_DIR):
        os.system("rm -rv "+OUTPUT_DIR)
        print("*** Deleting old output logs")
    os.system("mkdir "+OUTPUT_DIR)

    acc_configs  = json.load(open(ACC_CONFIGS))
    array_dims   = acc_configs['array_dims']
    i_sram_sizes = acc_configs['i_sram_sizes']
    w_sram_sizes = acc_configs['w_sram_sizes']
    o_sram_sizes = acc_configs['o_sram_sizes']
        
    topologies = sorted(os.listdir(TOPOLOGIES_DIR))
    print(topologies)

    experiment_num = 0

    for topology in topologies:
        for acc_num in range(len(array_dims)):
            print("********** Running Experiment {}/{} **********".format(experiment_num+1, len(topologies)*len(array_dims)*3))
            gen_config(topology, acc_num, array_dims[acc_num], i_sram_sizes[acc_num], w_sram_sizes[acc_num], o_sram_sizes[acc_num], "is")
            run_config(topology, acc_num, "is")
            experiment_num+=1
            print("********** Running Experiment {}/{} **********".format(experiment_num+1, len(topologies)*len(array_dims)*3))
            gen_config(topology, acc_num, array_dims[acc_num], i_sram_sizes[acc_num], w_sram_sizes[acc_num], o_sram_sizes[acc_num], "ws")
            run_config(topology, acc_num, "ws")
            experiment_num+=1
            print("********** Running Experiment {}/{} **********".format(experiment_num+1, len(topologies)*len(array_dims)*3))
            gen_config(topology, acc_num, array_dims[acc_num], i_sram_sizes[acc_num], w_sram_sizes[acc_num], o_sram_sizes[acc_num], "os")
            run_config(topology, acc_num, "os")
            experiment_num+=1
