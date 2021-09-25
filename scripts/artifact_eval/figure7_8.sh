dlrm_pt_bin="python RecPipeMain.py"

DATA_DIR="/data/rec/criteo_artifact//"

data_set="kaggle"
raw_data_file="$DATA_DIR/train.txt"
pre_processedd_file="$DATA_DIR/kaggleAdDisplayChallenge_processed.npz"
data="--data-generation=dataset --data-set=$data_set --raw-data-file=$raw_data_file --processed-data-file=$pre_processedd_file"


args="$data"
echo $args

outfile="logs/figure7_8_kaggle1.out"
touch $outfile
date >> $outfile
$dlrm_pt_bin $args $dlrm_extra_option --recpipe_configs "configs/recpipe_configs/kaggle_1_stage.json" | tee -a $outfile
date >> $outfile

outfile="logs/figure7_8_kaggle2.out"
touch $outfile
date >> $outfile
$dlrm_pt_bin $args $dlrm_extra_option --recpipe_configs "configs/recpipe_configs/kaggle_2_stage.json" | tee -a $outfile
date >> $outfile

outfile="logs/figure7_8_kaggle3.out"
touch $outfile
date >> $outfile
$dlrm_pt_bin $args $dlrm_extra_option --recpipe_configs "configs/recpipe_configs/kaggle_3_stage.json" | tee -a $outfile
date >> $outfile
