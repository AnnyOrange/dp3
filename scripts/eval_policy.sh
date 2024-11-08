# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0

#!/bin/bash

DEBUG=False
alg_name=${1}
addition_info=${2}
seed=${3}
gpu_id=${4}

# 定义你要执行评估的任务列表 "metaworld_soccer" "metaworld_shelf-place" "metaworld_sweep-into" metaworld_stick-push 
tasks=("metaworld_pick-out-of-hole" "metaworld_push" "metaworld_shelf-place" "metaworld_soccer" "metaworld_stick-push" "metaworld_sweep-into")

# 循环遍历每一个任务并执行评估
for task_name in "${tasks[@]}"; do
    exp_name="${task_name}-${alg_name}-${addition_info}"
    run_dir="data/outputs/${exp_name}_seed${seed}"

    echo -e "\033[33mRunning evaluation for task: ${task_name} with alg: ${alg_name} and seed: ${seed}\033[0m"
    echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

    cd 3D-Diffusion-Policy

    export HYDRA_FULL_ERROR=1 
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    python eval.py --config-name=${alg_name}.yaml \
                   task=${task_name} \
                   hydra.run.dir=${run_dir} \
                   training.debug=$DEBUG \
                   training.seed=${seed} \
                   training.device="cuda:0" \
                   exp_name=${exp_name} \
                   logging.mode="online" \
                   checkpoint.save_ckpt=True
    
    cd ..
done
