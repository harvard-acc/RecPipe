dlrm_pt_bin="python RecPipeMain.py"

# ####################################################
# MovieLens1m
# ####################################################
data_set="movielens1m"
DATA_DIR="/group/vlsiarch/ugupta/git/MSR/Neumf_Inference/dataset/"
data="--data-generation=dataset --data-set=$data_set  --processed-data-file=$DATA_DIR"

# ####################################################
# MovieLens20m
# ####################################################
#data_set="movielens20m"
#DATA_DIR="/group/vlsiarch/jpombra/20m_models"
#data="--data-generation=dataset --data-set=$data_set  --processed-data-file=$DATA_DIR"

args="$data"
echo $args
$dlrm_pt_bin $args $dlrm_extra_option --recpipe_configs "configs/recpipe_configs/movielens1m_1_stage.json"

