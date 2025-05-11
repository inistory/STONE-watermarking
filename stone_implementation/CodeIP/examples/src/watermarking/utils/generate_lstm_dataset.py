import os
import io
import torch
import tokenize
import argparse
import re
from torch.utils.data import Dataset, DataLoader
from torch import nn
from datasets import load_dataset
from typing import List

TOKEN_TYPE_TO_ID = {}
SEQ_LEN = 30

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

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

def tokenize_java_code(code: str) -> List[int]:
    try:
        token_ids = []
        # Java token types
        token_types = {
            'KEYWORD': r'\b(public|private|protected|class|interface|extends|implements|static|final|void|int|String|boolean|return|if|else|while|for|try|catch|throw|throws|new|this|super)\b',
            'IDENTIFIER': r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            'NUMBER': r'\b\d+\b',
            'STRING': r'"[^"]*"',
            'OPERATOR': r'[+\-*/=<>!&|^%]',
            'SEPARATOR': r'[(){}[\];,.]',
            'COMMENT': r'//.*|/\*[\s\S]*?\*/',
            'WHITESPACE': r'\s+'
        }
        
        # Create a pattern that matches any token
        pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_types.items())
        
        for match in re.finditer(pattern, code):
            token_type = match.lastgroup
            if token_type == 'WHITESPACE' or token_type == 'COMMENT':
                continue
            if token_type not in TOKEN_TYPE_TO_ID:
                TOKEN_TYPE_TO_ID[token_type] = len(TOKEN_TYPE_TO_ID)
            token_ids.append(TOKEN_TYPE_TO_ID[token_type])
            
        return token_ids
    except Exception as e:
        print(f"⚠️ Tokenize error: {e}")
        return []

class TypeSeqDataset(Dataset):
    def __init__(self, seq_len: int = SEQ_LEN, sample_size: int = 1000, dataset_type: str = "code_search_net", language: str = "python"):
        self.samples = []
        
        dataset = load_dataset("code_search_net", language, split="train")
        code_key = "func_code_string"
        doc_key = "func_documentation_string"
            
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

        print(f"\n🔍 sample key list: {dataset.column_names}")
        print(f"📌 used keys: '{code_key}', '{doc_key}'")

        for i in range(3):
            print(f"\n[{i}] {code_key}:\n{dataset[i][code_key][:300]}")
            print(f"\n[{i}] {doc_key}:\n{dataset[i][doc_key][:300]}")

        # Select tokenizer based on language
        tokenizer = {
            'python': tokenize_python_code,
            'java': tokenize_java_code
        }.get(language, tokenize_python_code)

        for idx, item in enumerate(dataset):
            code = item.get(code_key)
            if not code:
                continue
            type_ids = tokenizer(code)
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
    parser = argparse.ArgumentParser(description='Generate LSTM dataset')
    parser.add_argument('--dataset_type', type=str, default='code_search_net',
                      help='Type of dataset to use (e.g., code_search_net, humanevalpack)')
    parser.add_argument('--language', type=str, default='python',
                      help='Programming language to use (e.g., python, java)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for training')
    args = parser.parse_args()
    
    # Set generated_length based on dataset_type
    generated_length = 512 if args.dataset_type == "humanevalpack" else SEQ_LEN
    
    # Create dataset
    dataset = TypeSeqDataset(dataset_type=args.dataset_type, language=args.language)
    print(f"\n📦 Loaded {len(dataset)} samples.")
    print(f"📚 Vocab size (token types): {len(TOKEN_TYPE_TO_ID)}")
    print(f"📏 Generated length: {generated_length}")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(vocab_size=len(TOKEN_TYPE_TO_ID)).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("\n🚀 Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, len(TOKEN_TYPE_TO_ID)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save the model
    model_path = f"lstm_model_{args.language}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': TOKEN_TYPE_TO_ID,
        'seq_len': generated_length
    }, model_path)
    print(f"\n✅ Model saved to {model_path}")
