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
from evalplus.data import get_human_eval_plus, get_mbpp_plus

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        embedded = self.embedding(x)
        out, (hn, cn) = self.lstm(embedded, (h0, c0))
        output = self.fc(hn[-1])
        return output

def main(args: WmBaseArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(args.device)
    lm_tokenizer = tokenizer

    model_path = os.path.join(ROOT_PATH, "watermarking", "utils", f"lstm_model_{args.language}.pth")
    state_dict = torch.load(model_path, map_location=args.device)
    vocab_size = state_dict['embedding.weight'].shape[0]
    embed_size = 64
    hidden_size = 128
    output_size = vocab_size

    lstm_model = LSTMModel(vocab_size, embed_size, hidden_size, output_size)
    lstm_model.load_state_dict(state_dict)
    lstm_model.to(args.device)
    
    # Load dataset based on dataset type
    if args.dataset_type == "humaneval":
        dataset_dict = get_human_eval_plus()
    elif args.dataset_type == "mbpp":
        dataset_dict = get_mbpp_plus()
    elif args.dataset_type == "humanevalpack":
        dataset = []
        # Construct path based on language
        if args.language not in ["cpp", "java"]:
            raise ValueError(f"Unsupported language for humanevalpack: {args.language}. Supported languages are: cpp, java")
            
        output_file = os.path.join(ROOT_PATH, 'humanevalpack', 'data', args.language, 'data', 'humanevalpack.jsonl')
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Dataset file not found: {output_file}")
            
        with open(output_file, 'r') as f:
            for line in f:
                problem = json.loads(line)
                dataset.append({
                    'task_id': problem['task_id'],
                    'prompt': problem['prompt'],
                    'canonical_solution': problem['canonical_solution']
                })
        dataset_dict = {d['task_id']: d for d in dataset}
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    dataset = list(dataset_dict.values())[:args.sample_num]  # Convert dict to list and limit samples

    texts = [d['prompt'] + d['canonical_solution'] for d in dataset]

    lm_message_model = RandomMessageModel(tokenizer=tokenizer,
                                          lm_tokenizer=lm_tokenizer,
                                          delta=args.delta,
                                          message_code_len=args.message_code_len,
                                          device=model.device)

    watermark_processor = WmProcessorRandomMessageModel(message_model=lm_message_model,
                                                        tokenizer=tokenizer,
                                                        encode_ratio=args.encode_ratio,
                                                        message=args.message,
                                                        top_k=args.top_k)

    eos_id = torch.tensor(tokenizer.eos_token_id).to(model.device)
    min_length_processor = MinLengthLogitsProcessor(min_length=10000, eos_token_id=eos_id)

    rep_processor = RepetitionPenaltyLogitsProcessor(penalty=args.repeat_penalty)
    ngram_processor = NoRepeatNGramLogitsProcessor(ngram_size=args.ngram_size)

    pda_model = PDAMessageModel(tokenizer=tokenizer, pda_model=lstm_model, delta=args.delta)
    pda_model.lexer = PythonLexer()

    pda_processor = PDAProcessorMessageModel(message_model=pda_model,
                                             tokenizer=tokenizer,
                                             gamma=args.gamma)

    logit_processor = LogitsProcessorList([
        min_length_processor,
        rep_processor,
        ngram_processor,
        watermark_processor,
        pda_processor
    ])

    results = {
        'text': [],
        'prefix_and_output_text': [],
        'output_text': [],
        'decoded_message': [],
        'acc': []
    }

    try:
        for text in tqdm(texts):
            try:
                print(f"\n🔍 Processing text: {text[:100]}...")  # Print first 100 chars
                
                tokenized_input = tokenizer(text, return_tensors='pt')
                tokenized_input = truncate(tokenized_input, max_length=args.prompt_length)
                tokenized_input = tokenized_input.to(model.device)
                
                print(f"📊 Tokenized input shape: {tokenized_input['input_ids'].shape}")

                watermark_processor.start_length = tokenized_input['input_ids'].shape[-1]

                with torch.no_grad():
                    try:
                        print("🚀 Starting generation...")
                        output_tokens = model.generate(
                            **tokenized_input,
                            temperature=1.0,  # Set to 1.0 since we handle temperature manually
                            max_new_tokens=args.generated_length,
                            num_beams=args.num_beams,
                            repetition_penalty=None,
                            logits_processor=logit_processor,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            do_sample=True
                        )
                        print(f"✅ Generation completed. Output shape: {output_tokens.shape}")
                    except Exception as e:
                        print(f"❌ Error during generation: {e}")
                        continue

                try:
                    output_text = tokenizer.batch_decode(
                        output_tokens[:, tokenized_input["input_ids"].shape[-1]:],
                        skip_special_tokens=True
                    )[0]
                    print(f"📝 Generated output: {output_text[:100]}...")  # Print first 100 chars

                    prefix_and_output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
                    print(f"📝 Full text: {prefix_and_output_text[:100]}...")  # Print first 100 chars

                    results['text'].append(text)
                    results['output_text'].append(output_text)
                    results['prefix_and_output_text'].append(prefix_and_output_text)

                    decoded_message = watermark_processor.decode(output_text, disable_tqdm=True)[0]
                    available_message_num = args.generated_length // (int(args.message_code_len * args.encode_ratio))
                    acc = decoded_message[:available_message_num] == args.message[:available_message_num]

                    results['decoded_message'].append(decoded_message)
                    results['acc'].append(acc)

                    print(f"🎯 Decoded message: {decoded_message}, Accuracy: {acc}")
                except Exception as e:
                    print(f"❌ Error during post-processing: {e}")
                    continue

            except Exception as e:
                print(f"❌ Error processing text: {e}")
                continue

            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        pass

    args_dict = vars(args)
    results['args'] = args_dict
    results['task_id'] = [d['task_id'] for d in dataset]

    print(f"\n💾 Saving results to {args.save_path}")
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("✅ Results saved successfully")
