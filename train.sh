#!/bin/bash
trap "exit" INT
LOG_ROOT="$HOME/log/rat"
DATA_DIR="$HOME/data"
MODEL_DIR="$HOME/model"
NUM_WORKERS=6
TRAIN_SPLIT=0.5
VAL_SPLIT=0.2

# Setup attacks
attack_eps=$1; shift
# attack_eps='4'
if [[ $attack_eps = '4' ]]; then
    EPS='4/255' ALPHA='1/255' STEP=10
    EXP_NAME='eps4'
elif [[ $attack_eps = '3' ]]; then
    EPS='3/255' ALPHA='0.75/255' STEP=10
    EXP_NAME='eps3'
elif [[ $attack_eps = '8' ]]; then
    EPS='8/255' ALPHA='2/255' STEP=10
    EXP_NAME='eps8'
elif [[ $attack_eps = '12' ]]; then
    EPS='12/255' ALPHA='3/255' STEP=10
    EXP_NAME='eps12'
elif [[ $attack_eps = '16' ]]; then
    EPS='16/255' ALPHA='4/255' STEP=10
    EXP_NAME='eps16'
else
    echo "invalid eps: $attack_eps"
    exit 1
fi
echo "eps: $EPS step: $STEP step_size: $ALPHA"

# Setup dataset-specific params
if [[ $1 = 'office' ]]; then
    LOG_DIR="$LOG_ROOT/office/"
    ds='office'
    sd_epoch=100; ad_epoch=100; wacv_epoch=10
    shot_lr='1e-2'; shot_lr_bb='1e-3'
elif [[ $1 = 'officehome' ]]; then
    LOG_DIR="$LOG_ROOT/officehome/"
    ds='office_home'
    sd_epoch=50; ad_epoch=100; wacv_epoch=10
    shot_lr='1e-2'; shot_lr_bb='1e-3'
elif [[ $1 = 'pacs' ]]; then
    LOG_DIR="$LOG_ROOT/pacs/"
    ds='pacs'
    sd_epoch=50; ad_epoch=100; wacv_epoch=10
    shot_lr='1e-2'; shot_lr_bb='1e-3'
elif [[ $1 = 'visda' ]]; then
    LOG_DIR="$LOG_ROOT/visda/"
    ds='visda'
    sd_epoch=10; ad_epoch=50; wacv_epoch=5
    shot_lr='1e-3'; shot_lr_bb='1e-4'
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

# Architecture
# ARCH=$1; shift
if [[ $ds = 'visda' ]]; then
    # ARCH=$1; shift
    ARCH='resnet101'
else
    ARCH='resnet50'
fi
EXP_NAME="$EXP_NAME-$ARCH"

# Setup pre-training
if [[ $1 = 'std' ]]; then
    if [[ $ARCH = 'resnet50' ]]; then
        PT="$MODEL_DIR/resnet50-19c8e357.pth"
    elif [[ $ARCH = 'resnet101' ]]; then
        PT="$MODEL_DIR/resnet101-5d3b4d8f.pth"
    fi
elif [[ $1 = 'rob4' ]]; then
    PT="$MODEL_DIR/resnet50-imagenet_linf_4.pth"
elif [[ $1 = 'rob8' ]]; then
    PT="$MODEL_DIR/resnet50-imagenet_linf_8.pth"
else
    echo "invalid pre-training: $1"
    exit 1
fi
PT_NAME=$1; shift
echo "exp_name: $EXP_NAME"
echo "pre-training: $PT"


# Functions for running different main Python scripts
data_params="--data_dir $DATA_DIR --log_dir $LOG_DIR --num_workers $NUM_WORKERS --train_split $TRAIN_SPLIT --val_split $VAL_SPLIT"
shot_src () {
    python ./main_standard.py \
        $data_params \
        --arch $ARCH --arch_variant shot --dim 256 \
        --optimizer sgd_nesterov --weight_decay 1e-3 \
        --epoch $sd_epoch --batch 64 --lr $shot_lr --lr_bb $shot_lr_bb --lr_schedule dao \
        --label_smooth \
        --log_period 5 --ckpt_period 10000 --test_period 10 \
        "$@"
}
shot () {
    python ./main_shot.py  \
        $data_params \
        --arch $ARCH --arch_variant shot --dim 256 \
        --optimizer sgd_nesterov --weight_decay 1e-3 \
        --epoch 15 --batch 64 --lr $shot_lr --lr_bb $shot_lr_bb --lr_schedule dao \
        --log_clean_acc \
        --log_period 5 --ckpt_period 10000 --test_period 10 \
        "$@"
}
sup () {
    python ./main_standard.py \
        $data_params \
        --arch $ARCH --arch_variant shot --dim 256 \
        --log_clean_acc \
        --log_period 5 --ckpt_period 10000 --test_period 10 \
        "$@"
}
distill () {
    python ./main_distill.py \
        $data_params \
        --arch $ARCH --arch_variant shot --dim 256 \
        --log_clean_acc \
        --log_period 5 --ckpt_period 10000 --test_period 10 \
        "$@"
}

# Setup arguments
echo "args: $@"
methods=$1; shift
alpha=$1; shift
seed=$1; shift

attack_params="--attack_eps $EPS --attack_alpha $ALPHA --attack_steps $STEP --eval_attack pgd --eval_attack_eps $EPS --eval_attack_alpha $ALPHA --eval_attack_steps $STEP"
# attack_params="--attack_eps $EPS --attack_alpha $ALPHA --attack_steps $STEP --eval_attack pgd --eval_attack_eps 8/255 --eval_attack_alpha 1/255 --eval_attack_steps 20"
ad_optim="--optimizer sgd --lr 1e-1 --lr_bb 1e-1 --lr_schedule cos --epoch $ad_epoch"
number="--load_run_number $seed --teacher_run_number $seed --run_number $seed --seed $seed"

# Run
for task in $tasks; do
    s=${task:0:1}; t=${task:2:1}
    shot_teacher="$task-SHOT"

    for method in $methods; do
        # Source model training
        if [[ $method = 'SHOT_source' ]]; then
            shot_src --exp_name $EXP_NAME \
                --run_name "$task-SHOT_src" \
                --dataset "${ds}_$s" --target_dataset "${ds}_$t" \
                --pretrained $PT \
                $number "$@"
        # Standard source-free UDA 
        elif [[ $method = 'SHOT' ]]; then
            shot --exp_name $EXP_NAME \
                --run_name $shot_teacher \
                --dataset "${ds}_$t" \
                --load_run_name "$task-SHOT_src" \
                $number "$@"
        # Adversarial training (hard pseudo labels)
        elif [[ $method = 'SHOT_PGDAT' ]]; then
            sup --exp_name $EXP_NAME \
                --run_name "$task-${PT_NAME}_PT-SHOT-PGD_AT" \
                --dataset "${ds}_$t" \
                --pretrained $PT \
                --teacher_run_name $shot_teacher \
                --at_mode pgd_at --attack pgd $attack_params \
                $ad_optim $number "$@"
        # Relaxed Adversarial Training (RAT)
        elif [[ $method = 'SHOT_RAT' ]]; then
            distill --exp_name $EXP_NAME \
                --run_name "$task-${PT_NAME}_PT-SHOT-RAT_${alpha}" \
                --dataset "${ds}_$t" \
                --pretrained $PT \
                --teacher_run_name $shot_teacher \
                --at_mode rat --rat_alpha $alpha \
                --attack soft_pgd $attack_params \
                $ad_optim $number "$@"
        fi
    done
done

echo "END."


