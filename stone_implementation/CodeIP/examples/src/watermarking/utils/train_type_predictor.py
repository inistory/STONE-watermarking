import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from generate_lstm_dataset import TypeSeqDataset
import os
import argparse

EMBED_SIZE = 64
HIDDEN_SIZE = 128
SEQ_LENGTH = 30
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.001

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='python',
                      choices=['python', 'java', 'cpp'],
                      help='Programming language')
    return parser.parse_args()

class TypePredictor(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TypePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        logits = self.fc(out)
        return logits

def train():
    args = parse_args()
    
    # Set model path based on language
    if args.language == 'java':
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_model_java.pth")
        vocab_size = 100
    elif args.language == 'cpp':
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_model_cpp.pth")
        vocab_size = 100
    else:  # python
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_model_python.pth")
        vocab_size = 79

    dataset = TypeSeqDataset(seq_len=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"📚 Using vocab size: {vocab_size} for {args.language}")

    model = TypePredictor(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"🧠 Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
