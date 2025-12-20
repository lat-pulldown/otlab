import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import math
import joblib

# --- CONFIGURATION DEFAULTS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CYBER_MODEL_PATH = 'model/hybrid_model.pth'
PHYS_MODEL_PATH = 'model/hybrid_multivar.pth'
SCALER_PATH = 'model/multivar_scaler.pkl'

# Model Settings
WINDOW_SIZE = 10
NUM_CLASSES = 100 
HIDDEN_SIZE = 64
PHYS_FEATURES = 3 

# --- SHARED MODULES ---
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

# --- CYBER MODEL ---
class HybridModel(nn.Module):
    def __init__(self, num_classes, hidden_size, window_size, nhead=4, num_layers=2):
        super(HybridModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=window_size + 10)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * window_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1) 
        x = self.fc(x)
        return x

# --- PHYSICAL MODEL ---
class VariateModel(nn.Module):
    def __init__(self, input_dim, d_model, window_size, nhead=4, num_layers=2):
        super(VariateModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size + 10)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model * window_size, input_dim) 

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = x.permute(0, 2, 1) 
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# --- DYNAMIC DATA LOADING ---
def load_data(cyber_file, phys_file):
    print(f"Loading Cyber Data: {cyber_file}...")
    try:
        path_c = cyber_file if os.path.exists(cyber_file) else f'data/{cyber_file}'
        df_c = pd.read_csv(path_c, header=None, names=['eid', 'label'])
    except:
        print(f"Error: {cyber_file} not found.")
        exit()

    seq_c = [x - 1 for x in df_c['eid'].values.tolist()]
    labels = df_c['label'].values.tolist()

    print(f"Loading Physical Data: {phys_file}...")
    try:
        path_p = phys_file if os.path.exists(phys_file) else f'data/{phys_file}'
        df_p = pd.read_csv(path_p)
    except:
        print(f"Error: {phys_file} not found.")
        exit()
        
    vals_p = df_p[['eid_count', 'eid_nunique', 'temperature']].values.astype(np.float32)

    # --- SCALER LOADING ---
    if os.path.exists(SCALER_PATH):
        print(f"Loading Scaler from {SCALER_PATH}...")
        scaler = joblib.load(SCALER_PATH)
        vals_p = scaler.transform(vals_p)
        vals_p = np.nan_to_num(vals_p)
    else:
        print("WARNING: Scaler not found! Results will be inaccurate.")

    min_len = min(len(seq_c), len(vals_p))
    print(f"Synchronizing datasets... Testing on {min_len - WINDOW_SIZE} windows.")
    return seq_c[:min_len], vals_p[:min_len], labels[:min_len]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # New Arguments
    parser.add_argument('-cyber', default='mix01.csv', help="Cyber log dataset filename")
    parser.add_argument('-phys', default='mix_tf01.csv', help="Physical sensor dataset filename")
    
    # Tuning Arguments
    parser.add_argument('-k', default=2, type=int, help="Top-K for Cyber Model")
    parser.add_argument('-mse', default=0.352157, type=float, help="MSE Threshold for Physical Model")
    args = parser.parse_args()

    seq_c, vals_p, labels = load_data(args.cyber, args.phys)

    print("Loading Models...")
    cyber_model = HybridModel(NUM_CLASSES, HIDDEN_SIZE, WINDOW_SIZE).to(DEVICE)
    cyber_model.load_state_dict(torch.load(CYBER_MODEL_PATH, map_location=DEVICE))
    cyber_model.eval()

    phys_model = VariateModel(PHYS_FEATURES, HIDDEN_SIZE, WINDOW_SIZE).to(DEVICE)
    phys_model.load_state_dict(torch.load(PHYS_MODEL_PATH, map_location=DEVICE))
    phys_model.eval()

    print(f"Running Fusion (Cyber K={args.k}, Phys MSE={args.mse})...")
    
    tp_or, fp_or, fn_or, tn_or = 0, 0, 0, 0
    tp_and, fp_and, fn_and, tn_and = 0, 0, 0, 0

    with torch.no_grad():
        for i in range(len(seq_c) - WINDOW_SIZE):
            # --- CYBER ---
            pattern_c = torch.tensor([seq_c[i:i+WINDOW_SIZE]], dtype=torch.long).to(DEVICE)
            target_c = seq_c[i+WINDOW_SIZE]
            out_c = cyber_model(pattern_c)
            top_preds = torch.argsort(out_c, 1, descending=True)[0, :args.k].tolist()
            cyber_anomaly = target_c not in top_preds

            # --- PHYSICAL ---
            pattern_p = torch.tensor([vals_p[i:i+WINDOW_SIZE]], dtype=torch.float).to(DEVICE)
            target_p = torch.tensor([vals_p[i+WINDOW_SIZE]], dtype=torch.float).to(DEVICE)
            out_p = phys_model(pattern_p)
            loss_val = torch.mean((out_p - target_p) ** 2).item()
            phys_anomaly = loss_val > args.mse

            # --- FUSION ---
            is_anomaly_or = cyber_anomaly or phys_anomaly
            is_anomaly_and = cyber_anomaly and phys_anomaly
            truth = labels[i+WINDOW_SIZE]

            if truth == 1:
                tp_or += 1 if is_anomaly_or else 0
                fn_or += 1 if not is_anomaly_or else 0
                tp_and += 1 if is_anomaly_and else 0
                fn_and += 1 if not is_anomaly_and else 0
            else:
                fp_or += 1 if is_anomaly_or else 0
                tn_or += 1 if not is_anomaly_or else 0
                fp_and += 1 if is_anomaly_and else 0
                tn_and += 1 if not is_anomaly_and else 0

    def print_metrics(tp, fp, fn, tn, name):
        p = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
        r = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        acc = 100 * (tp + tn) / (tp + tn + fp + fn)
        print(f"\n--- {name} FUSION ---")
        print(f"Accuracy:  {acc:.3f}%")
        print(f"Precision: {p:.3f}%")
        print(f"Recall:    {r:.3f}%")
        print(f"F1-Score:  {f1:.3f}%")
        print(f"False Pos: {fp}")

    print("="*40)
    print("FUSION RESULTS")
    print("="*40)
    print_metrics(tp_or, fp_or, fn_or, tn_or, "OR (High Recall)")
    print_metrics(tp_and, fp_and, fn_and, tn_and, "AND (High Precision)")