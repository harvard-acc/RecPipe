dlrm_pt_bin="python RecPipeMain.py"

DATA_DIR="/data/rec/criteo_artifact//"

data_set="kaggle"
raw_data_file="$DATA_DIR/train.txt"
pre_processedd_file="$DATA_DIR/kaggleAdDisplayChallenge_processed.npz"
data="--data-generation=dataset --data-set=$data_set --raw-data-file=$raw_data_file --processed-data-file=$pre_processedd_file"


args="$data"
echo $args

$dlrm_pt_bin $args $dlrm_extra_option --recpipe_configs "configs/recpipe_configs/test.json"


