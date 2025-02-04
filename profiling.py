import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3" 
import torch
import numpy as np
from datasets import load_dataset
import pickle
from tqdm import tqdm
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def get_long_contexts():
    subsets = ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh', 'hotpotqa', '2wikimqa', 'musique', 'dureader', 'gov_report', 'qmsum', 'multi_news', 'vcsum', 'trec', 'triviaqa', 'samsum', 'lsht', 'passage_count', 'passage_retrieval_en', 'passage_retrieval_zh', 'lcc', 'repobench-p', 'qasper_e', 'multifieldqa_en_e', 'hotpotqa_e', '2wikimqa_e', 'gov_report_e', 'multi_news_e', 'trec_e', 'triviaqa_e', 'samsum_e', 'passage_count_e', 'passage_retrieval_en_e', 'lcc_e', 'repobench-p_e']
    if not os.path.exists('long_contexts.pkl'):
        long_contexts = []
        for subset in subsets:
            print(subset)
            dataset = load_dataset('THUDM/LongBench', subset, split='test')
            for example in dataset:
                context = example['context']
                if len(context) > 15000:
                    long_contexts.append(context[:15000])

        with open('long_contexts.pkl', 'wb') as f:
            pickle.dump(long_contexts, f)

    with open('long_contexts.pkl', 'rb') as f:
        long_contexts = pickle.load(f)
    long_contexts = random.sample(long_contexts, 6000)
    long_contexts = random.sample(long_contexts, 200)
    return long_contexts

def get_expert_collaboration(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_experts = model.config.num_experts

    long_contexts = get_long_contexts()
    collaboration_all = np.zeros((n_layers, n_experts, n_experts), dtype=int)
    for batch in tqdm(range(0, len(long_contexts), 2)):
        inputs = tokenizer(long_contexts[batch:batch+2], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        model(**inputs, return_dict=True)
        for layer in range(n_layers):
            if 'Qwen' in model.config.architectures[0]:
                top_indices = model.model.layers[layer].mlp.top_k_indices.cpu().numpy()
            elif hasattr(model.model.layers[layer].mlp, 'gate'):
                top_indices = model.model.layers[layer].mlp.gate.top_k_indices.cpu().numpy()
            else:
                break
            for i, experts in enumerate(top_indices): 
                collaboration_all[layer, experts[:, None], experts] += 1
                            
    collaboration_all[..., np.arange(n_experts), np.arange(n_experts)] = -1
    return collaboration_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='llama-moe/LLaMA-MoE-v1-3_5B-2_8')
    parser.add_argument('--top_t', type=int, default=5)
    args = parser.parse_args()

    path = args.model_path
    top_t = args.top_t

    expert_collaboration_all = get_expert_collaboration(path)
    with open(f'topt_{top_t}_collaboration.pkl', 'wb') as f:
        pickle.dump(expert_collaboration_all, f)

    collaborative_list = np.argsort(-expert_collaboration_all, axis=2)
    collaborative_list = collaborative_list.tolist()
    with open(f'topt_{top_t}_collaborative_list.pkl', 'wb') as f:
        pickle.dump(collaborative_list, f)