import torch
from watermark.auto_watermark import AutoWatermark, STONEAutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import pickle 
import json
from custom_evalplus.evalplus.data import get_mbpp_plus, get_human_eval_plus
import sklearn.metrics as metrics
import numpy as np
import sys
import os
from dotenv import load_dotenv
            
def get_roc_aur(human_z, machine_z):
    assert len(human_z) == len(machine_z)

    baseline_z_scores = np.array(human_z)
    watermark_z_scores = np.array(machine_z)
    all_scores = np.concatenate([baseline_z_scores, watermark_z_scores])

    baseline_labels = np.zeros_like(baseline_z_scores)
    watermarked_labels = np.ones_like(watermark_z_scores)
    all_labels = np.concatenate([baseline_labels, watermarked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    return roc_auc, fpr, tpr, thresholds


def calculate_perplexity(text, tokenizer, model):
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity


def calculate_average_perplexity(texts, tokenizer, model):
    perplexities = [calculate_perplexity(str(text), tokenizer, model) for text in texts]
    average_perplexity = sum(perplexities) / len(perplexities)
    return average_perplexity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Watermarking methods
    parser.add_argument('--method', type=str, default="STONE")
    
    # Default parameters
    parser.add_argument('--prefix_length', type=int, default=1)
    parser.add_argument('--z_threshold', type=float, default=10.0) 
    parser.add_argument('--f_scheme', type=str, default="time")
    parser.add_argument('--window_scheme', type=str, default="left")
    parser.add_argument('--entropy_threshold', type=float, default=0.9)

    parser.add_argument('--data', type=str, default="mbppplus")
    parser.add_argument('--model', type=str, default="llama")
    # Hyperparameters 
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.5)
    parser.add_argument('--hash_key', type=int, default=15485863)

    # Watermarking strategy
    parser.add_argument('--skipping_rule', type=str, default="all_pl")
    parser.add_argument('--watermark_on_pl', type=str, default="False")
    
    #etc
    parser.add_argument('--language', type=str, default="python")  
    parser.add_argument('--n_samples', type=int, default=1)
    
    args = parser.parse_args()
    
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_access_token = os.getenv("HF_ACCESS_TOKEN")
    os.environ["HF_ACCESS_TOKEN"] = hf_access_token

    if args.data == "humanevalpack":
        max_new_tokens_length = 512
    else:
        max_new_tokens_length = 200

    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct").to(device),
                                            tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct"),
                                            vocab_size=50272,
                                            device=device,
                                            max_new_tokens=max_new_tokens_length,
                                            min_length=200,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)
    # Load watermark algorithm
    if args.method == "STONE":
        myWatermark = STONEAutoWatermark.load(args.method, 
                                        transformers_config=transformers_config, 
                                        skipping_rule=args.skipping_rule, watermark_on_pl=args.watermark_on_pl, 
                                        gamma=args.gamma, delta=args.delta, hash_key=args.hash_key, 
                                        z_threshold=args.z_threshold, prefix_length=args.prefix_length,
                                        language=args.language) 
    else:
        myWatermark = AutoWatermark.load(args.method, 
                                        transformers_config=transformers_config, 
                                        gamma=args.gamma, 
                                        delta=args.delta, 
                                        hash_key=args.hash_key, 
                                        z_threshold=args.z_threshold, 
                                        prefix_length=args.prefix_length, 
                                        f_scheme=args.f_scheme, 
                                        window_scheme=args.window_scheme, 
                                        entropy_threshold=args.entropy_threshold)
        
    codes = []
    prompts = []
    task_ids = []
    data = []
    example_tests = []
    tests = []
    if args.data == "mbppplus":
        for task_id, problem in get_mbpp_plus().items():
            codes.append(problem['canonical_solution'])
            prompts.append(problem['prompt'])
            task_ids.append(task_id)
    elif args.data == "humanevalplus":
        for task_id, problem in get_human_eval_plus().items():
            codes.append(problem['prompt'] + problem['canonical_solution'])
            prompts.append(problem['prompt'])
            task_ids.append(task_id)
    elif args.data == "humanevalpack":
        output_file = f'./custom_evalplus/humanevalpack/data/{args.language}/data/humanevalpack.jsonl'
        with open(output_file, 'r') as f:
            for line in f:
                problem = json.loads(line)
                task_id = problem['task_id']
                codes.append(problem['prompt'] + problem['canonical_solution'])
                prompts.append(problem['prompt'])
                task_ids.append(task_id)
                example_tests.append(problem['example_test'])
                tests.append(problem['test'])
    else:
        print("Data not supported")
        sys.exit(1)

    # watermarked llm-generated codes
    watermarked_codes = []
    # watermarked llm code z-score
    watermarked_detect_results = []
    # unwatermarked llm-generated codes 
    unwatermarked_codes = []
    # unwatermarked llm code z-score
    unwatermarked_detect_results = []
    # human code z-score
    human_code_detect_results = []    
    #perplexity
    average_perplexity_watermarked = []
    average_perplexity_unwatermarked = []   

    # setting model for perplexity calculation
    model_name = "bigcode/starcoder2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
       
    for task_id, prompt, code in zip(task_ids, prompts, codes):
        watermarked_completions = []
        unwatermarked_completions = []
        
        detect_result = myWatermark.detect_watermark(code)
        human_code_detect_results.append(detect_result['score'])
        
        watermarked_code = myWatermark.generate_watermarked_text(prompt)
        detect_result = myWatermark.detect_watermark(watermarked_code)
        watermarked_completions.append(watermarked_code)
        watermarked_detect_results.append(detect_result['score'])
        
        unwatermarked_code = myWatermark.generate_unwatermarked_text(prompt)
        detect_result = myWatermark.detect_watermark(unwatermarked_code)
        unwatermarked_completions.append(unwatermarked_code)
        unwatermarked_detect_results.append(detect_result['score'])

        # Perplexity
        watermarked_perplexity = calculate_perplexity(watermarked_code, tokenizer, model)
        unwatermarked_perplexity = calculate_perplexity(unwatermarked_code, tokenizer, model)
        average_perplexity_watermarked.append(watermarked_perplexity)
        average_perplexity_unwatermarked.append(unwatermarked_perplexity)

        if args.data == 'humanevalpack':
            watermarked_codes.append(watermarked_completions)
            unwatermarked_codes.append(unwatermarked_completions)  

        else:   
            watermarked_codes.append({"task_id": task_id, "completion": watermarked_completions})
            unwatermarked_codes.append({"task_id": task_id, "completion": unwatermarked_completions})                   

    # Calculate average perplexity
    average_perplexity_watermarked = sum(average_perplexity_watermarked) / len(average_perplexity_watermarked)
    average_perplexity_unwatermarked = sum(average_perplexity_unwatermarked) / len(average_perplexity_unwatermarked)     
        
    # evaluated results
    human_zscore = np.mean(human_code_detect_results)
    watermarked_zscore = np.mean(watermarked_detect_results)
    unwatermarked_zscore = np.mean(unwatermarked_detect_results)
    
    roc_aur_list = []
    fpr_list = []
    tpr_list = []
    thresholds_list = []

    roc_aur, fpr, tpr, thresholds = get_roc_aur(human_code_detect_results, watermarked_detect_results)
    roc_aur_list.append(roc_aur)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    thresholds_list.append(thresholds)

    eval_results = {}
    eval_results['human_detect_results'] = human_code_detect_results
    eval_results['watermarked_detect_results'] = watermarked_detect_results
    eval_results['unwatermarked_detect_results'] = unwatermarked_detect_results
    eval_results['human_zscore'] = human_zscore
    eval_results['watermarked_zscore'] = watermarked_zscore
    eval_results['unwatermarked_zscore'] = unwatermarked_zscore

    # Detection performance - Human Vs. Watermarked
    eval_results['roc_aur-human-watermarked'] = roc_aur_list
    eval_results['fpr-human-watermarked'] = fpr_list
    eval_results['tpr-human-watermarked'] = tpr_list
    eval_results['thresholds-human-watermarked'] = thresholds_list
    
    # Perplexity results
    eval_results['average_perplexity_watermarked'] = average_perplexity_watermarked
    eval_results['average_perplexity_unwatermarked'] = average_perplexity_unwatermarked



    watermarked_solutions = [] 
    if args.data == "humanevalpack":
        for task_id, codes in zip(task_ids, watermarked_codes):
            watermarked_solutions.append(codes)
    elif args.data == "mbppplus":
        for watermarked_code in watermarked_codes:
            watermarked_solutions.append({'task_id': watermarked_code['task_id'], 'solution': watermarked_code['completion']})
    else:
        for watermarked_code in watermarked_codes:
            watermarked_solutions.append({'task_id': watermarked_code['task_id'], 'completion': watermarked_code['completion']})

    save_path = f'./results/{args.n_samples}samples/'


    os.makedirs(save_path, exist_ok=True)
    # save evaluation watermarked results 
    with open(save_path + args.data + '_' + args.language + '_' + args.model + '_' + args.method + '_' + args.skipping_rule + '_' + str(args.watermark_on_pl) + '_' + str(args.gamma) + '_' + str(args.delta) + '_' + str(args.hash_key) + '_' + '_evaluation_results.pickle', 'wb') as f:
        pickle.dump(eval_results, f)
    
    if args.data == "humanevalpack":
        with open(save_path + args.data + '_' + args.language + '_' + args.model + '_' + args.method + '_' + args.skipping_rule + '_' + str(args.watermark_on_pl) + '_' + str(args.gamma) + '_' + str(args.delta) + '_' + str(args.hash_key) + '_' + '_watermarked_solutions.json', 'w') as f:
            json.dump(watermarked_solutions, f, indent=4)        
    else:
        with open(save_path + args.data + '_' + args.language + '_' + args.model + '_' + args.method + '_' + args.skipping_rule + '_' + str(args.watermark_on_pl) + '_' + str(args.gamma) + '_' + str(args.delta) + '_' + str(args.hash_key) + '_' + '_watermarked_solutions.jsonl', 'w') as f:
            for item in watermarked_solutions:
                f.write(json.dumps(item) + '\n')
            
            
    # save evaluation unwatermarked results 
    unwatermarked_solutions = [] 
    if args.data == "humanevalpack":
        for task_id, codes in zip(task_ids, unwatermarked_codes):
            unwatermarked_solutions.append(codes)
    elif args.data == "mbppplus":
        for unwatermarked_code in unwatermarked_codes:
            unwatermarked_solutions.append({'task_id': unwatermarked_code['task_id'], 'solution': unwatermarked_code['completion']})
    else:
        for unwatermarked_code in unwatermarked_codes:
            unwatermarked_solutions.append({'task_id': unwatermarked_code['task_id'], 'completion': unwatermarked_code['completion']})

    # save unwatermarked solutions
    if args.data == "humanevalpack":
        with open(save_path + args.data + '_' + args.language + '_' + args.model + '_' + args.method + '_' + str(args.hash_key) + '_' + '_unwatermarked_solutions.json', 'w') as f:
            json.dump(unwatermarked_solutions, f, indent=4)
    else:
        with open(save_path + args.data + '_' + args.language + '_' + args.model + '_' + args.method +'_' + str(args.hash_key) + '_' + '_unwatermarked_solutions.jsonl', 'w') as f:
            for item in unwatermarked_solutions:
                f.write(json.dumps(item) + '\n')