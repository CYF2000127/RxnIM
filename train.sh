#!/bin/bash
#SBATCH --job-name=cyf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=5
#SBATCH --mem=500gb
#SBATCH --partition=gpu3090
#SBATCH --gres=gpu:10
#SBATCH --output=/home/ychenkv/shikra-main/finetune_4.121_detr_notemp.out  
#module load your_module

accelerate launch --num_processes 10 \
    mllm/pipeline/finetune.py \
    config/shikra_pretrain_final19_stage2.py \
    --cfg-options model_args.model_name_or_path=llama-7b \
    --per_device_train_batch_size 2 \
    --output_dir ./exp/reaction_4.12_detr_notemp \
    
    #--overwrite_output_dir ./exp/reaction_4 \
    #--main_process_port 23786 \
