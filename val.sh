#!/bin/bash
#SBATCH --job-name=cyf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=5
#SBATCH --mem=500gb
#SBATCH --partition=gpu3090
#SBATCH --gres=gpu:10
#SBATCH --output=/home/ychenkv/shikra-main/finetune_val.out  

#module load your_module

accelerate launch --num_processes 10 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_multi_pope.py \
        --cfg-options model_args.model_name_or_path=./shikra-7b \
        --per_device_eval_batch_size 1