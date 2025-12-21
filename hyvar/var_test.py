import torch
import torch.nn as nn
import time
import argparse
import pandas as pd
import numpy as np
import psutil
import os
import math
import joblib
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *** CONFIGURATION ***
# Updated threshold based on your successful tuner run
OPTIMAL_THRESHOLD = 0.352157 
FEATURE_COLS = ['eid_count', 'eid_nunique', 'temperature']
SCALER_PATH = 'model/multivar_scaler.pkl'
MODEL_PATH = 'model/hybrid_multivar.pth'

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

    # Load Scaler
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        data = scaler.transform(data)
        data = np.nan_to_num(data)
    else:
        print("WARNING: Scaler not found! Results will be inaccurate.")

    inputs, targets, truth = [], [], []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i + window_size])
        targets.append(data[i + window_size])
        truth.append(labels[i + window_size])
        
    return np.array(inputs), np.array(targets), np.array(truth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='../data/mix_tf.csv')
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-threshold', default=OPTIMAL_THRESHOLD, type=float)
    args = parser.parse_args()

    inputs, targets, labels = load_data(args.data, args.window_size)
    if len(inputs) == 0: exit()

    model = HybridMultivarModel(len(FEATURE_COLS), args.hidden_size, args.window_size).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Error: Model {MODEL_PATH} not found.")
        exit()
        
    model.eval()
    
    print(f"Starting evaluation with MSE Threshold > {args.threshold}...")
    
    # --- PERFORMANCE METRICS SETUP ---
    process = psutil.Process(os.getpid())
    start_time = time.time()
    latencies = []
    y_pred = []
    
    with torch.no_grad():
        # Iterate one by one for latency measurement (or small batches)
        # To match your hybrid report's latency style, we do per-batch timing
        batch_size = 512
        for i in range(0, len(inputs), batch_size):
            t0 = time.time()
            
            batch_in = torch.tensor(inputs[i:i+batch_size], dtype=torch.float).to(device)
            batch_target = torch.tensor(targets[i:i+batch_size], dtype=torch.float).to(device)
            
            output = model(batch_in)
            
            # Calculate MSE
            loss = torch.mean((output - batch_target) ** 2, dim=1)
            
            # Apply Threshold
            anomalies = (loss > args.threshold).long().cpu().numpy()
            y_pred.extend(anomalies)
            
            t1 = time.time()
            # Approx latency per sample = batch time / batch size
            batch_latency = (t1 - t0) * 1000 # ms
            latencies.extend([batch_latency / len(batch_in)] * len(batch_in))

    total_time = time.time() - start_time
    
    # --- RESOURCE & TIMING METRICS ---
    end_ram = process.memory_info().rss / 1024 / 1024 # MB
    cpu_usage = process.cpu_percent()
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = len(inputs) / total_time
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    
    acc = 100 * (tp + tn) / len(labels)
    p = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    r = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    fpr = 100 * fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # --- FINAL REPORT ---
    print('=' * 40)
    print(f"  EVALUATION REPORT - MULTIVARIATE HYBRID")
    print('=' * 40)
    print(f"CONFUSION MATRIX:")
    print(f"  TP: {tp:<5} | FP: {fp:<5}")
    print(f"  FN: {fn:<5} | TN: {tn:<5}")
    print('-' * 40)
    print(f"ACCURACY METRICS:")
    print(f"Accuracy:      {acc:.3f}%")
    print(f"-----")
    print(f"Precision:     {p:.3f}%")
    print(f"Recall:        {r:.3f}%")
    print(f"F1 Score:      {f1:.3f}%")
    print(f"FPR:           {fpr:.3f}%")
    print('-' * 40)
    print(f"PERFORMANCE METRICS:")
    print(f"Total Time:    {total_time:.2f}s")
    print(f"Throughput:    {throughput:.0f} samples/sec")
    print(f"Avg Latency:   {avg_latency:.4f} ms")
    print(f"P99 Latency:   {p99_latency:.4f} ms")
    print(f"CPU Usage:     {cpu_usage:.1f}%")
    print(f"RAM Usage:     {end_ram:.1f} MB")
    print('=' * 40)