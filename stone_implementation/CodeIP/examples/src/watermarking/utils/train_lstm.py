
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

LANGUAGE = "python"
VOCAB_SIZE = 79
SEQ_LENGTH = 30
EMBED_SIZE = 64
HIDDEN_SIZE = 128
OUTPUT_SIZE = VOCAB_SIZE
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = f"lstm_model_{LANGUAGE}.pth"

class RandomSequenceDataset(Dataset):
    def __init__(self, num_samples=10000):
        self.data = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LENGTH))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        logits = self.fc(out)
        return logits

def train():
    dataset = RandomSequenceDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
