import json
import os.path
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor
from .arg_classes.wm_arg_class import WmBaseArgs
from .watermark_processors.message_model_processor import WmProcessorRandomMessageModel
from .watermark_processors.message_models.message_model import RandomMessageModel
from .watermark_processors.PDA_model_processor import PDAProcessorMessageModel
from .watermark_processors.message_models.PDA_message_model import PDAMessageModel
from pygments.lexers import PythonLexer
import torch.nn as nn
from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
HUMANEVALPACK_PATH = os.path.join(ROOT_PATH, "humanevalpack")

def get_dataset(dataset_type, language="python"):
    if dataset_type == "humaneval":
        return get_human_eval_plus()
    elif dataset_type == "mbpp":
        return get_mbpp_plus()
    elif dataset_type == "humanevalpack":
        if language not in ["java", "cpp"]:
            raise ValueError(f"Unsupported language for HumanEvalPack: {language}")
        
        # Load from local JSONL file
        jsonl_path = os.path.join(HUMANEVALPACK_PATH, "data", language, "data", "humanevalpack.jsonl")
        formatted_dataset = {}
        
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                formatted_dataset[str(i)] = {
                    'task_id': str(i),
                    'prompt': item['prompt'],
                    'canonical_solution': item['canonical_solution']
                }
        return formatted_dataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d

def main(args: WmBaseArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(args.device)
    lm_tokenizer = tokenizer

    tokenizer.model_max_length = args.prompt_length

    model_path = os.path.join(os.path.dirname(__file__), "utils", f"lstm_model_{args.language}.pth")
    
    try:
        state_dict = torch.load(model_path, weights_only=True, map_location=args.device)
        
        if 'embedding.weight' not in state_dict:
            weight_mapping = {
                'lstm.weight_ih_l0': 'lstm.weight_ih_l0',
                'lstm.weight_hh_l0': 'lstm.weight_hh_l0',
                'lstm.bias_ih_l0': 'lstm.bias_ih_l0',
                'lstm.bias_hh_l0': 'lstm.bias_hh_l0',
                'fc.weight': 'fc.weight',
                'fc.bias': 'fc.bias'
            }
            
            for key in state_dict.keys():
                if 'embed' in key.lower():
                    weight_mapping[key] = 'embedding.weight'
                    vocab_size = state_dict[key].shape[0]
                    break
            else:
                print("Warning: Could not find embedding weights, using default vocabulary size")
                vocab_size = tokenizer.vocab_size
        
        else:
            vocab_size = state_dict['embedding.weight'].shape[0]
            weight_mapping = None

        embed_size = 64
        hidden_size = 128
        output_size = vocab_size

        lstm_model = LSTMModel(vocab_size, embed_size, hidden_size, output_size)
        
        if weight_mapping:
            new_state_dict = {}
            for old_key, new_key in weight_mapping.items():
                if old_key in state_dict:
                    new_state_dict[new_key] = state_dict[old_key]
            lstm_model.load_state_dict(new_state_dict, strict=False)
        else:
            lstm_model.load_state_dict(state_dict)
            
        lstm_model.to(args.device)
        print(f"Successfully loaded LSTM model with vocabulary size: {vocab_size}")
        
    except Exception as e:
        print(f"Error loading LSTM model: {str(e)}")
        print("Falling back to random message model only")
        args.use_pda = False  
        lstm_model = None

    dataset = get_dataset(args.dataset_type, args.language)
    dataset = list(dataset.values())[:args.sample_num]

    texts = [d['prompt'] for d in dataset]
    canonical_solutions = [d['canonical_solution'] for d in dataset]  

    lm_message_model = RandomMessageModel(
        tokenizer=tokenizer,
        lm_tokenizer=lm_tokenizer,
        delta=args.delta,
        message_code_len=args.message_code_len,
        device=model.device
    )

    if args.use_pda and lstm_model is not None:
        pda_model = PDAMessageModel(
            tokenizer=tokenizer, 
            pda_model=lstm_model, 
            delta=args.delta
        )
        pda_model.lexer = PythonLexer()
        watermark_processor = PDAProcessorMessageModel(
            message_model=pda_model,
            tokenizer=tokenizer,
            gamma=args.gamma,
            beta=args.beta
        )
    else:
        watermark_processor = WmProcessorRandomMessageModel(
            message_model=lm_message_model,
            tokenizer=tokenizer,
            encode_ratio=args.encode_ratio,
            message=args.message,
            top_k=args.top_k
        )

    eos_id = torch.tensor(tokenizer.eos_token_id).to(model.device)
    
    logit_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(
            min_length=min(1000, args.generated_length),
            eos_token_id=eos_id
        ),
        NoRepeatNGramLogitsProcessor(ngram_size=args.ngram_size),
        watermark_processor 
    ])

    results = {
        'text': [],
        'prefix_and_output_text': [],
        'output_text': [],
        'decoded_message': [],
        'acc': [],
        'canonical_solution': [],
        'task_id': []
    }

    try:
        for text, canonical_solution, task_id in tqdm(zip(texts, canonical_solutions, [d['task_id'] for d in dataset])):
            for _ in range(args.n_samples):
                tokenized_input = tokenizer(
                    text, 
                    return_tensors='pt',
                    truncation=True,
                    max_length=args.prompt_length
                ).to(model.device)

                watermark_processor.start_length = tokenized_input['input_ids'].shape[-1]

                output_tokens = model.generate(
                    **tokenized_input,
                    temperature=args.temperature,
                    max_new_tokens=args.generated_length,
                    num_beams=1,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=args.repeat_penalty,
                    logits_processor=logit_processor
                )

                output_text = tokenizer.batch_decode(
                    output_tokens[:, tokenized_input["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )[0]

                prefix_and_output_text = tokenizer.batch_decode(
                    output_tokens, 
                    skip_special_tokens=True
                )[0]

                decoded_message = watermark_processor.decode(output_text, disable_tqdm=True)[0]
                message_ratio = args.message_code_len * args.encode_ratio
                available_message_num = int(args.generated_length / message_ratio)  # 정수 나눗셈 대신 float 사용
                acc = decoded_message[:available_message_num] == args.message[:available_message_num]

                results['text'].append(text)
                results['output_text'].append(output_text)
                results['prefix_and_output_text'].append(prefix_and_output_text)
                results['canonical_solution'].append(canonical_solution)
                results['task_id'].append(task_id)
                results['decoded_message'].append(decoded_message)
                results['acc'].append(acc)

                print(f"Task ID: {task_id}")
                print(f"Generated Code:\n{output_text}")
                print(f"Decoded Message: {decoded_message}")
                print(f"Accuracy: {acc}\n")

                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        pass

    args_dict = vars(args)
    results['args'] = args_dict
    results['extraction_rate'] = sum(results['acc']) / len(results['acc']) if results['acc'] else 0

    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)  
        output = self.fc(output[:, -1, :]) 
        return output
