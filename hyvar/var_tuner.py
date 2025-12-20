import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import math
import joblib
from sklearn.metrics import precision_recall_fscore_support

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_COLS = ['eid_count', 'eid_nunique', 'temperature']
SCALER_PATH = 'model/multivar_scaler.pkl'
MODEL_PATH = 'model/hybrid_multivar.pth'
WINDOW_SIZE = 10

# --- MODEL DEFINITION ---
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
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class HybridMultivarModel(nn.Module):
    def __init__(self, num_features, hidden_size, window_size, nhead=4, num_layers=2):
        super(HybridMultivarModel, self).__init__()
        self.input_proj = nn.Linear(num_features, hidden_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=window_size + 10)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size * window_size, num_features)

    def forward(self, x):
        x = self.input_proj(x)
        x_cnn = self.cnn(x.permute(0, 2, 1))
        x_trans = self.pos_encoder(x_cnn.permute(0, 2, 1))
        x_trans = self.transformer_encoder(x_trans)
        x_flat = x_trans.contiguous().view(x_trans.size(0), -1)
        return self.fc(x_flat)

def load_data(filename, window_size):
    print(f"Loading {filename}...")
    try:
        path = filename if os.path.exists(filename) else f'data/{filename}'
        df = pd.read_csv(path)
    except:
        print(f"Error: {filename} not found.")
        return [], [], []

    data = df[FEATURE_COLS].values.astype(np.float32)
    labels = df['label'].values.astype(int)
    
    # LOAD SCALER
    if os.path.exists(SCALER_PATH):
        print(f"Loading Scaler from {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        data = scaler.transform(data)
        data = np.nan_to_num(data)
    else:
        print(f"ERROR: {SCALER_PATH} not found.")
        return [], [], []
    
    inputs, targets, truth = [], [], []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i + window_size])
        targets.append(data[i + window_size])
        truth.append(labels[i + window_size])
    return np.array(inputs), np.array(targets), np.array(truth)

def tune(model, inputs, targets, labels):
    print("Calculating MSE scores...")
    model.eval()
    mse_scores = []
    
    with torch.no_grad():
        for i in range(0, len(inputs), 512):
            batch_in = torch.tensor(inputs[i:i+512], dtype=torch.float).to(DEVICE)
            batch_target = torch.tensor(targets[i:i+512], dtype=torch.float).to(DEVICE)
            output = model(batch_in)
            loss = torch.mean((output - batch_target) ** 2, dim=1)
            mse_scores.extend(loss.cpu().numpy())
    
    mse_scores = np.array(mse_scores)
    print(f"MSE Stats -> Min: {mse_scores.min():.6f} | Max: {mse_scores.max():.6f} | Mean: {mse_scores.mean():.6f}")

    # --- UPDATED SWEEP STRATEGY (Percentiles) ---
    print("\nSweeping thresholds using percentiles (Zooming in on the boundary)...")
    # Sweep from 0 to 99.5 percentile to avoid the massive 13000 outlier
    candidates = np.unique(np.percentile(mse_scores, np.linspace(0, 99.5, 300)))
    
    best_f1, best_thresh = 0, 0
    best_metrics = (0, 0)
    
    print(f"{'Threshold':<12} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    
    for thresh in candidates:
        preds = (mse_scores > thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
            best_metrics = (p, r)
            print(f"{thresh:<12.6f} | {f1:<10.2%} | {p:<10.2%} | {r:<10.2%}")
            
    print("-" * 55)
    print(f"WINNER: Optimal MSE Threshold = {best_thresh:.6f}")
    print(f"Stats:  F1: {best_f1:.2%} (Prec: {best_metrics[0]:.2%}, Rec: {best_metrics[1]:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='mix_tf01.csv')
    args = parser.parse_args()

    inputs, targets, labels = load_data(args.data, WINDOW_SIZE)
    if len(inputs) == 0: exit()

    model = HybridMultivarModel(len(FEATURE_COLS), 64, WINDOW_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    tune(model, inputs, targets, labels)