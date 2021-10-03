#!/usr/bin/env bash

set -exu

sweep_id=$1
partition=$2
max_run=${3:-100}
num_machines=${4:-0}
mem=${5:-25000}
threads=${6:-1}


TIME=`(date +%Y-%m-%d-%H-%M-%S-%N)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

model_name="wandb"
dataset=$sweep_id
job_name="$model_name-$dataset-$TIME"
log_dir=logs/$model_name/$dataset/$TIME
log_base=$log_dir/log

partition=$partition

mkdir -p $log_dir

sbatch -J $job_name \
            -e $log_base.err \
            -o $log_base.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --gres=gpu:1 \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --array=0-$num_machines \
            --exclude=node026,node040,node057,node059,node072,node095,node099,node123,node125,node167,node169,node176 \
            bin/run_sweep.sh $sweep_id $threads $max_run

