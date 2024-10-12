#!/bin/bash

DEBUG=False
save_ckpt=True
two_times=True  # Renamed variable

alg_name=${1}
addition_info=${2}
seed=${3}
gpu_id=${4}

tasks=("metaworld_pick-place-wall" "metaworld_push" "metaworld_soccer" "metaworld_sweep-into" "metaworld_stick-pull")

for task_name in "${tasks[@]}"; do
    exp_name="${task_name}-${alg_name}-${addition_info}"
    run_dir="data/outputs/${exp_name}_seed${seed}_2x"

    echo -e "\033[33mRunning task: ${task_name} with alg: ${alg_name} and seed: ${seed}\033[0m"
    echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

    if [ $DEBUG = True ]; then
        wandb_mode=offline
        echo -e "\033[33mDebug mode!\033[0m"
    else
        wandb_mode=online
        echo -e "\033[33mTrain mode\033[0m"
    fi

    cd 3D-Diffusion-Policy

    export HYDRA_FULL_ERROR=1 
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    # 调用 Python 脚本进行训练
    python train.py --config-name=${alg_name}.yaml \
                    task=${task_name} \
                    hydra.run.dir=${run_dir} \
                    training.debug=$DEBUG \
                    training.seed=${seed} \
                    training.device="cuda:0" \
                    training.two_times=${two_times} \
                    exp_name=${exp_name} \
                    logging.mode=${wandb_mode} \
                    checkpoint.save_ckpt=${save_ckpt}
                    
    cd ..
done
