# The official implementation of Advancing MoE Efficiency: A Collaboration-Constrained Routing ($\texttt{C2R}$) Strategy for Better Expert Parallelism Design.

## Requirements

For **LLaMA-MoE**: please refer to the [official repo](https://github.com/pjlab-sys4nlp/llama-moe) for conda environment installation.

For **Qwen-MoE**:
```bash
conda create -n c2r python=3.11
conda activate c2r
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate deepspeed wandb huggingface_hub
```

## Reproduce the results

Take **Qwen-MoE** for example:
1. Download the weights of `Qwen/Qwen1.5-MoE-A2.7B` from Huggingface locally. You can use [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) if encounter the network issue. Replace `{YOUR HF_TOKEN}` with your Huggingface token.
    ```bash
    huggingface-cli download --token {YOUR HF_TOKEN} --resume-download Qwen/Qwen1.5-MoE-A2.7B--local-dir ./Qwen1.5-MoE-A2.7B --local-dir-use-symlinks False
    ```
2. Run the following command for profiling:
    ```python
    python profiling.py --model_path Qwen/Qwen1.5-MoE-A2.7B top_t 30
    ```
    
    It will generate three files:
    * `long_contexts.pkl`: the corpus for profiling.
    * `topt_30_collaboration.pkl`: the **Expert Collaboration Matrix** for each layer.
    * `topt_30_collaborative_list.pkl`: the **Top-T list** for each layer.
3. Manually add the **top_t** and **collaborative_list** to the config file of model in the downloaded model folder. See `./Qwen-MoE/model_topt30/config.json` for example which is used in the paper.
4. Replace the modeling file in the downloaded model folder with our revised modeling file in `./Qwen-MoE/model_topt30/modeling_qwen2_moe.py` which implements the **C2R** strategy. Search `@revise` in the file for the revised part.
5. Download the [LIMA](https://huggingface.co/datasets/GAIR/lima) dataset train the model with the script in `./Qwen-MoE/train.sh`:
    ```bash
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
    ```
5. Evaluate with the official [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repo.

## File structure

```
.
├── collaboration_data/: Expert Collaboration Matrix of LLaMA-MoE for reproduce the results in the paper
├── LLaMA-MoE/
│   ├── config.json: add `top_t` and `collaborative_list` to the config file of LLaMA-MoE
│   ├── modeling_llama_moe_hf.py: revised modeling file for LLaMA-MoE, mainly for `TopKBalancedNoisyGate` class
│   ├── train.sh: script for sft. Revised from `https://github.com/pjlab-sys4nlp/llama-moe/blob/main/scripts/sft/2_8.sh`
├── Qwen-MoE/
│   ├── configs/: configuration files for finetuning
│   ├── model_topt30/: configuration files for Qwen-MoE
│   │   ├── config.json: add `top_t` and `collaborative_list` to the config file of Qwen-MoE
│   │   ├── modeling_qwen2_moe.py: revised modeling file for Qwen-MoE, mainly for `Qwen2MoeSparseMoeBlock` class
│   ├── finetune.py: finetune script for Qwen-MoE
│   ├── train.sh
└── profiling.py: profiling script
```