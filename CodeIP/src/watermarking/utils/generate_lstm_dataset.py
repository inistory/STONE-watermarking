import os
import io
import torch
import tokenize
import argparse
import re
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List

TOKEN_TYPE_TO_ID = {}
SEQ_LEN = 30

# 토큰 타입 정의
TOKEN_TYPES = {
    # 공통 토큰 타입
    'KEYWORD': 'keyword',
    'IDENTIFIER': 'identifier',
    'OPERATOR': 'operator',
    'SEPARATOR': 'separator',
    'LITERAL': 'literal',
    'COMMENT': 'comment',
    'WHITESPACE': 'whitespace',
    
    # C++ 특화 토큰 타입
    'CPP_PREPROCESSOR': 'cpp_preprocessor',
    'CPP_TEMPLATE': 'cpp_template',
    
    # Java 특화 토큰 타입
    'JAVA_ANNOTATION': 'java_annotation',
    'JAVA_MODIFIER': 'java_modifier'
}

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
    try:
        token_ids = []
        
        # C++ 키워드 목록
        cpp_keywords = {
            'auto', 'break', 'case', 'catch', 'class', 'const', 'continue', 'default',
            'delete', 'do', 'else', 'enum', 'explicit', 'export', 'extern', 'false',
            'for', 'friend', 'goto', 'if', 'inline', 'mutable', 'namespace', 'new',
            'operator', 'private', 'protected', 'public', 'return', 'sizeof', 'static',
            'struct', 'switch', 'template', 'this', 'throw', 'true', 'try', 'typedef',
            'typeid', 'typename', 'union', 'using', 'virtual', 'void', 'volatile', 'while'
        }
        
        # 연산자와 구분자
        operators = {'+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='}
        separators = {'(', ')', '[', ']', '{', '}', ';', ',', '.', ':', '::', '->', '.*', '->*'}
        
        # 코드를 라인별로 분리
        lines = code.split('\n')
        
        for line in lines:
            # 전처리기 지시문 처리
            if line.strip().startswith('#'):
                if 'CPP_PREPROCESSOR' not in TOKEN_TYPE_TO_ID:
                    TOKEN_TYPE_TO_ID['CPP_PREPROCESSOR'] = len(TOKEN_TYPE_TO_ID)
                token_ids.append(TOKEN_TYPE_TO_ID['CPP_PREPROCESSOR'])
                continue
            
            # 주석 처리
            if '//' in line:
                if 'COMMENT' not in TOKEN_TYPE_TO_ID:
                    TOKEN_TYPE_TO_ID['COMMENT'] = len(TOKEN_TYPE_TO_ID)
                token_ids.append(TOKEN_TYPE_TO_ID['COMMENT'])
                continue
            
            # 템플릿 처리
            if '<' in line and '>' in line:
                if 'CPP_TEMPLATE' not in TOKEN_TYPE_TO_ID:
                    TOKEN_TYPE_TO_ID['CPP_TEMPLATE'] = len(TOKEN_TYPE_TO_ID)
                token_ids.append(TOKEN_TYPE_TO_ID['CPP_TEMPLATE'])
            
            # 토큰 분리
            tokens = re.findall(r'\w+|[^\w\s]', line)
            
            for token in tokens:
                if token in cpp_keywords:
                    if 'KEYWORD' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['KEYWORD'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['KEYWORD'])
                elif token in operators:
                    if 'OPERATOR' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['OPERATOR'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['OPERATOR'])
                elif token in separators:
                    if 'SEPARATOR' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['SEPARATOR'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['SEPARATOR'])
                elif token.isdigit() or (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
                    if 'LITERAL' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['LITERAL'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['LITERAL'])
                elif token.strip():
                    if 'IDENTIFIER' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['IDENTIFIER'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['IDENTIFIER'])
        
        return token_ids
    except Exception as e:
        print(f"⚠️ C++ Tokenize error: {e}")
        return []

def tokenize_java_code(code: str) -> List[int]:
    try:
        token_ids = []
        
        # Java 키워드 목록
        java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
            'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
            'extends', 'final', 'finally', 'float', 'for', 'if', 'implements', 'import',
            'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private',
            'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch',
            'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile',
            'while'
        }
        
        # Java 수정자
        java_modifiers = {'public', 'private', 'protected', 'static', 'final', 'abstract', 'synchronized', 'volatile', 'transient', 'native'}
        
        # 연산자와 구분자
        operators = {'+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '>>>', '++', '--', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='}
        separators = {'(', ')', '[', ']', '{', '}', ';', ',', '.', ':', '::', '->'}
        
        # 코드를 라인별로 분리
        lines = code.split('\n')
        
        for line in lines:
            # 주석 처리
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                if 'COMMENT' not in TOKEN_TYPE_TO_ID:
                    TOKEN_TYPE_TO_ID['COMMENT'] = len(TOKEN_TYPE_TO_ID)
                token_ids.append(TOKEN_TYPE_TO_ID['COMMENT'])
                continue
            
            # 어노테이션 처리
            if '@' in line:
                if 'JAVA_ANNOTATION' not in TOKEN_TYPE_TO_ID:
                    TOKEN_TYPE_TO_ID['JAVA_ANNOTATION'] = len(TOKEN_TYPE_TO_ID)
                token_ids.append(TOKEN_TYPE_TO_ID['JAVA_ANNOTATION'])
            
            # 토큰 분리
            tokens = re.findall(r'\w+|[^\w\s]', line)
            
            for token in tokens:
                if token in java_keywords:
                    if 'KEYWORD' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['KEYWORD'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['KEYWORD'])
                elif token in java_modifiers:
                    if 'JAVA_MODIFIER' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['JAVA_MODIFIER'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['JAVA_MODIFIER'])
                elif token in operators:
                    if 'OPERATOR' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['OPERATOR'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['OPERATOR'])
                elif token in separators:
                    if 'SEPARATOR' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['SEPARATOR'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['SEPARATOR'])
                elif token.isdigit() or (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
                    if 'LITERAL' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['LITERAL'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['LITERAL'])
                elif token.strip():
                    if 'IDENTIFIER' not in TOKEN_TYPE_TO_ID:
                        TOKEN_TYPE_TO_ID['IDENTIFIER'] = len(TOKEN_TYPE_TO_ID)
                    token_ids.append(TOKEN_TYPE_TO_ID['IDENTIFIER'])
        
        return token_ids
    except Exception as e:
        print(f"⚠️ Java Tokenize error: {e}")
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
