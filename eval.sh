#!/bin/bash
trap "exit" INT
LOG_ROOT="$HOME/log/rat"
DATA_DIR="$HOME/data"
MODEL_DIR="$HOME/model"
NUM_WORKERS=6
TRAIN_SPLIT=0.5
VAL_SPLIT=0.2

# Setup attacks
# attack=$1; shift
attack='pgd'
attack_eps=$1; shift
if [[ $attack_eps = '4' ]]; then
    EPS='4/255' ALPHA='0.5/255' STEP=20
    EXP_NAME='eps4'
elif [[ $attack_eps = '3' ]]; then
    EPS='3/255' ALPHA='0.375/255' STEP=20
    EXP_NAME='eps3'
elif [[ $attack_eps = '8' ]]; then
    EPS='8/255' ALPHA='1/255' STEP=20
    EXP_NAME='eps8'
elif [[ $attack_eps = '12' ]]; then
    EPS='12/255' ALPHA='1.5/255' STEP=20
    EXP_NAME='eps12'
elif [[ $attack_eps = '16' ]]; then
    EPS='16/255' ALPHA='2/255' STEP=20
    EXP_NAME='eps16'
else
    echo "invalid eps: $attack_eps"
    exit 1
fi
attack_params="--attack $attack --attack_eps $EPS --attack_alpha $ALPHA --attack_steps $STEP"
echo "eps: $EPS step: $STEP step_size: $ALPHA"


# Setup dataset-specific params
if [[ $1 = 'office' ]]; then
    LOG_DIR="$LOG_ROOT/office/"
    ds='office'
elif [[ $1 = 'officehome' ]]; then
    LOG_DIR="$LOG_ROOT/officehome/"
    ds='office_home'
elif [[ $1 = 'pacs' ]]; then
    LOG_DIR="$LOG_ROOT/pacs/"
    ds='pacs'
elif [[ $1 = 'visda' ]]; then
    LOG_DIR="$LOG_ROOT/visda/"
    ds='visda'
else
    echo "invalid dataset: $1"
    exit 1
fi
shift
tasks=$1; shift
echo "dataset: $ds"
echo "tasks: $tasks"
echo "split: $TRAIN_SPLIT $VAL_SPLIT"
echo "log_dir: $LOG_DIR"

# Architecture and pre-training
# ARCH=$1; shift
if [[ $ds = 'visda' ]]; then
    # ARCH=$1; shift
    ARCH='resnet101'
else
    ARCH='resnet50'
fi
PT_NAME=$1; shift
EXP_NAME="$EXP_NAME-$ARCH"
echo "exp_name: $EXP_NAME"
echo "pre-training: $PT_NAME"

# Setup arguments
echo "args: $@"
methods=$1; shift
alphas=$1; shift
seed=$1; shift

# Function for running the Python script
data_params="--data_dir $DATA_DIR --log_dir $LOG_DIR --num_workers $NUM_WORKERS"
if [[ $methods = 'transductive' ]]; then
    data_params="$data_params --train_split none"
else
    data_params="$data_params --train_split $TRAIN_SPLIT --val_split $VAL_SPLIT"
fi
do_eval () {
    python ./eval.py \
        $data_params \
        --exp_name $EXP_NAME \
        --batch 64 \
        "$@"
}

# Run
dataset=''
load_run_name=''
load_run_number=''

for task in $tasks; do
    s=${task:0:1}; t=${task:2:1}

    for method in $methods; do
        for alpha in $alphas; do
            if [[ $method = 'SHOT_source' ]]; then
                load_name="$task-SHOT_src"
            elif [[ $method = 'SHOT' ]]; then
                load_name="$task-SHOT"
            elif [[ $method = 'SHOT_PGDAT' ]]; then
                load_name="$task-${PT_NAME}_PT-SHOT-PGD_AT"
            elif [[ $method = 'SHOT_RAT' ]]; then
                load_name="$task-${PT_NAME}_PT-SHOT-RAT_${alpha}"
            fi
            dataset="$dataset ${ds}_$t"
            load_run_name="$load_run_name $load_name"
            load_run_number="$load_run_number $seed"
        done
    done
done

do_eval --run_name "eval-$attack" \
    --dataset $dataset --load_run_name $load_run_name --load_run_number $load_run_number \
    $attack_params --seed $seed "$@"


echo "END."