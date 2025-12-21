import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import pandas as pd
import math

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *** CONFIGURATION ***
NUM_CLASSES = 100 
INPUT_FILE = '../data/train.csv' 

def generate(name, window_size):
    print(f"Loading {name}...")
    try:
        path = os.path.join('data', name)
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        try:
             df = pd.read_csv(name, header=None)
        except:
             print(f"Error: File {name} not found.")
             return None

    seq = df[0].values.tolist()
    seq = [n - 1 for n in seq] # Adjust for 0-based indexing

    inputs = []
    outputs = []
    
    for i in range(len(seq) - window_size):
        inputs.append(seq[i:i + window_size])
        outputs.append(seq[i + window_size])
        
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long))
    print(f"Total training samples: {len(inputs)}")
    return dataset

# --- MODEL SIZE PRINTER ---
def get_model_size(model):
    """Calculates and prints the model size in Parameters and MB."""
    param_count = sum(p.numel() for p in model.parameters())
    # 4 bytes per float32 parameter
    size_mb = param_count * 4 / (1024 * 1024)
    print('-' * 40)
    print(f"Model Structure Summary:")
    print(f"  Total Parameters: {param_count:,}")
    print(f"  Estimated Size:   {size_mb:.4f} MB")
    print('-' * 40)
    return size_mb

# --- POSITIONAL ENCODING ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Keep batch first: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- THE HYBRID MODEL ---
class HybridModel(nn.Module):
    def __init__(self, num_classes, hidden_size, window_size, nhead=4, num_layers=2):
        super(HybridModel, self).__init__()
        
        # 1. Embedding
        self.embedding = nn.Embedding(num_classes, hidden_size)
        
        # 2. Local Encoder: 1D-CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=window_size + 10)
        
        # 4. Global Reasoner: Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size*4, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 5. Output Head
        self.fc = nn.Linear(hidden_size * window_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x_cnn = x.permute(0, 2, 1) # Permute for CNN
        x_cnn = self.cnn(x_cnn) 
        x_trans = x_cnn.permute(0, 2, 1) # Permute back for Transformer
        x_trans = self.pos_encoder(x_trans)
        x_trans = self.transformer_encoder(x_trans)
        x_flat = x_trans.contiguous().view(x_trans.size(0), -1)
        out = self.fc(x_flat)
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_epochs', default=50, type=int)
    parser.add_argument('-batch_size', default=2048, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    args = parser.parse_args()

    model_dir = 'model'
    
    seq_dataset = generate(INPUT_FILE, args.window_size)
    if seq_dataset is None: exit()
        
    dataloader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    model = HybridModel(NUM_CLASSES, args.hidden_size, args.window_size).to(device)
    
    # --- PRINT MODEL SIZE ---
    get_model_size(model)
    # ------------------------
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training (Hybrid CNN-Transformer)... Device: {device}")
    start_time = time.time()
    
    for epoch in range(args.num_epochs): 
        model.train()
        train_loss = 0
        for seq, label in dataloader:
            seq = seq.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, train_loss / len(dataloader)))
    
    print('Total Time: {:.3f}s'.format(time.time() - start_time))
    
    if not os.path.isdir(model_dir): os.makedirs(model_dir)
    torch.save(model.state_dict(), f'{model_dir}/hybrid_model.pth')
    print('Model saved to model/hybrid_model.pth')