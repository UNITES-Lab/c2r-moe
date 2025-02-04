DATA_PATH="/path/to/lima"
OUTPUT_PATH="/path/to/output"
MODEL_PATH="/path/to/Qwen1.5-MoE-A2.7B"

deepspeed \
    --include localhost:0,1,2,3 \
    --master_addr localhost \
    --master_port 19858 \
finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 2 \
    --model_max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed /path/to/Qwen-MoE/configs/ds_config_zero3.json \
    --bf16 True \
    --use_lora False
