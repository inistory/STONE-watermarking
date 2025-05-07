import os
import io
import torch
import tokenize
import argparse
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List

TOKEN_TYPE_TO_ID = {}
SEQ_LEN = 30

def tokenize_python_code(code: str) -> List[int]:
    try:
        token_ids = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            tok_type = tokenize.tok_name.get(tok.type)
            if not tok_type:
                continue
            if tok_type not in TOKEN_TYPE_TO_ID:
                TOKEN_TYPE_TO_ID[tok_type] = len(TOKEN_TYPE_TO_ID)
            token_ids.append(TOKEN_TYPE_TO_ID[tok_type])
        return token_ids
    except Exception as e:
        print(f"⚠️ Tokenize error: {e}")
        return []

def tokenize_cpp_code(code: str) -> List[int]:
    # C++ 코드 토크나이징 로직 구현
    # TODO: C++ 코드 토크나이징 구현
    return []

def tokenize_java_code(code: str) -> List[int]:
    # Java 코드 토크나이징 로직 구현
    # TODO: Java 코드 토크나이징 구현
    return []

class TypeSeqDataset(Dataset):
    def __init__(self, language: str, seq_len: int = SEQ_LEN, sample_size: int = 1000):
        self.samples = []
        self.language = language.lower()
        
        # 언어별 데이터셋 로드
        if self.language == "python":
            dataset = load_dataset("code_search_net", "python", split="train")
            tokenize_func = tokenize_python_code
        elif self.language == "cpp":
            dataset = load_dataset("code_search_net", "cpp", split="train")
            tokenize_func = tokenize_cpp_code
        elif self.language == "java":
            dataset = load_dataset("code_search_net", "java", split="train")
            tokenize_func = tokenize_java_code
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

        print(f"\n🔍 sample key list: {dataset.column_names}")
        print(f"📌 used key: 'func_code_string'")

        for i in range(3):
            print(f"\n[{i}] func_code_string:\n{dataset[i]['func_code_string'][:300]}")

        for idx, item in enumerate(dataset):
            code = item.get("func_code_string")
            if not code:
                continue
            type_ids = tokenize_func(code)
            if len(type_ids) < seq_len:
                continue
            for i in range(len(type_ids) - seq_len):
                x = type_ids[i:i + seq_len - 1]
                y = type_ids[i + 1:i + seq_len]
                self.samples.append((torch.tensor(x), torch.tensor(y)))

        print(f"\n✅ token type vocab: {list(TOKEN_TYPE_TO_ID.keys())[:10]} ...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate LSTM dataset for code type prediction')
    parser.add_argument('--language', type=str, required=True, choices=['python', 'cpp', 'java'],
                      help='Programming language to process')
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN,
                      help='Sequence length for training')
    parser.add_argument('--sample_size', type=int, default=1000,
                      help='Number of samples to process')
    
    args = parser.parse_args()
    
    dataset = TypeSeqDataset(language=args.language, seq_len=args.seq_len, sample_size=args.sample_size)
    print(f"\n📦 Loaded {len(dataset)} samples.")
    print(f"📚 Vocab size (token types): {len(TOKEN_TYPE_TO_ID)}")
