#!/usr/bin/bash

export WANDB_PROJECT="llama_moe_sft"
num_gpus=4

{
    model_type="auto"

    task_name="llama_moe_2_8_deita"
    model_name_or_path="/path/to/LLaMA-MoE-v1-3_5B-2_8_topt_5"
    dataset_dir_or_path="/path/to/llama-moe/data/deita/deita_6k.jsonl"
    output_dir="/path/to/output"

    comment="llama-moe 2/8, deita, w/ balance loss, w/ freeze gate, w/ gate noise"
    mkdir -p $output_dir
    git diff > $output_dir/diff.patch
    env > $output_dir/env

    deepspeed \
    --include localhost:4,5,6,7 \
    --master_addr localhost \
    --master_port 12351 \
        /path/to/llama-moe/smoe/entrypoint/sft/train_sft.py \
            --do_train \
            --freeze_gate False \
            --evaluation_strategy no \
            --run_name $task_name \
            --model_type $model_type \
            --model_name_or_path $model_name_or_path \
            --dataset_dir_or_path $dataset_dir_or_path \
            --output_dir $output_dir \
            --deepspeed /path/to/llama-moe/conf/deepspeed/bf16_zero1.json \
            --seed 12306 \
            --bf16 True \
            --tf32 True \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --num_train_epochs 2 \
            --save_strategy steps \
            --save_steps 10 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --report_to wandb

}
