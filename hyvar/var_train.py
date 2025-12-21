import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import pandas as pd
import numpy as np
import math
import joblib
from sklearn.preprocessing import StandardScaler

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *** CONFIGURATION ***
INPUT_FILE = '../data/train_tf.csv' 
FEATURE_COLS = ['eid_count', 'eid_nunique', 'temperature']
WINDOW_SIZE = 10
SCALER_PATH = 'model/multivar_scaler.pkl'
MODEL_PATH = 'model/hybrid_multivar.pth'

def load_and_process_data(filename, window_size, is_train=True):
    path = filename if os.path.exists(filename) else os.path.join('data', filename)
    print(f"Loading {path}...")
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return None

    # Extract Features
    data = df[FEATURE_COLS].values.astype(np.float32)
    
    # Scaling (Critical for Multivariate data)
    if is_train:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        # Fix for columns with constant values (std=0) to avoid NaN
        data = np.nan_to_num(data) 
        if not os.path.exists('model'): os.makedirs('model')
        joblib.dump(scaler, SCALER_PATH)
    else:
        # Load scaler to ensure test data is scaled exactly like training data
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            data = scaler.transform(data)
            data = np.nan_to_num(data)
        else:
            print("Warning: Scaler not found. Using raw data.")

    # Create Sliding Windows
    # Input: Window of (t-W to t-1) -> Output: Vector at (t)
    inputs = []
    outputs = []
    
    for i in range(len(data) - window_size):
        inputs.append(data[i:i + window_size])
        outputs.append(data[i + window_size])
        
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.array(outputs), dtype=torch.float32)
    
    print(f"Generated {len(inputs)} windows. Input Shape: {inputs.shape}")
    return TensorDataset(inputs, outputs)

# --- MODEL COMPONENTS ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HybridMultivarModel(nn.Module):
    def __init__(self, num_features, hidden_size, window_size, nhead=4, num_layers=2):
        super(HybridMultivarModel, self).__init__()
        
        # 1. Feature Projection (Replaces Embedding)
        # Projects 3 features -> Hidden Size (64)
        self.input_proj = nn.Linear(num_features, hidden_size)
        
        # 2. Local Encoder: 1D-CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=window_size + 10)
        
        # 4. Global Reasoner: Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size*4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 5. Output Head (Predicts next vector)
        # Flatten [Batch, Window, Hidden] -> [Batch, Window*Hidden]
        self.fc = nn.Linear(hidden_size * window_size, num_features)

    def forward(self, x):
        # x: [Batch, Window, Features]
        
        x = self.input_proj(x) # -> [Batch, Window, Hidden]
        
        x_cnn = x.permute(0, 2, 1) # -> [Batch, Hidden, Window]
        x_cnn = self.cnn(x_cnn)
        
        x_trans = x_cnn.permute(0, 2, 1) # -> [Batch, Window, Hidden]
        x_trans = self.pos_encoder(x_trans)
        x_trans = self.transformer_encoder(x_trans)
        
        x_flat = x_trans.contiguous().view(x_trans.size(0), -1)
        out = self.fc(x_flat) # -> [Batch, Features]
        return out

def get_model_size(model):
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / (1024 * 1024)
    print('-' * 40)
    print(f"Model Structure Summary:")
    print(f"  Total Parameters: {param_count:,}")
    print(f"  Estimated Size:   {size_mb:.4f} MB")
    print('-' * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_epochs', default=50, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    args = parser.parse_args()

    # Load Data
    dataset = load_and_process_data(INPUT_FILE, args.window_size, is_train=True)
    if dataset is None: exit()
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Init Model
    num_features = len(FEATURE_COLS)
    model = HybridMultivarModel(num_features, args.hidden_size, args.window_size).to(device)
    
    get_model_size(model)
    
    # Loss: MSE for Continuous Data Forecasting
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training on {FEATURE_COLS}...")
    start_time = time.time()
    
    for epoch in range(args.num_epochs): 
        model.train()
        train_loss = 0
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], MSE Loss: {train_loss/len(dataloader):.6f}')
    
    print(f'Total Training Time: {time.time() - start_time:.3f}s')
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
    
    # Calculate Threshold (Max MSE on Training Data)
    model.eval()
    errors = []
    with torch.no_grad():
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            output = model(seq)
            # MSE per sample
            loss = torch.mean((output - target) ** 2, dim=1)
            errors.extend(loss.cpu().numpy())
            
    # Set threshold as Mean + 3*Std (or Max for strictness)
    threshold = np.mean(errors) + 3 * np.std(errors)
    joblib.dump(threshold, 'model2/multivar_threshold.pkl')
    print(f"Anomaly Threshold set to: {threshold:.6f}")