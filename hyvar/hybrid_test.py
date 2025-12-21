import torch
import torch.nn as nn
import time
import argparse
import pandas as pd
import os
import numpy as np
import psutil
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 100 

def generate_test_data(name, window_size):
    print(f"Loading {name}...")
    try:
        df = pd.read_csv('data/' + name, header=None, names=['eid', 'label'])
    except FileNotFoundError:
        try:
             df = pd.read_csv(name, header=None, names=['eid', 'label'])
        except:
             print(f"Error: File {name} not found in current directory or 'data/' folder.")
             return []
    
    seq = [x - 1 for x in df['eid'].values.tolist()]
    labels = df['label'].values.tolist()
    
    data = []
    for i in range(len(seq) - window_size):
        pattern = seq[i:i + window_size]
        target_eid = seq[i + window_size]
        target_label = labels[i + window_size]
        data.append((pattern, target_eid, target_label))
        
    print(f"Total test windows: {len(data)}")
    return data

# --- POSITIONAL ENCODING (FIXED TO MATCH TRAIN) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # The checkpoint was saved with unsqueeze(0) -> [1, max_len, d_model]
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [Batch, Seq, Hidden]
        return x + self.pe[:, :x.size(1)]

# --- HYBRID MODEL (FIXED TO MATCH TRAIN) ---
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
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size*4, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(hidden_size * window_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        
        # CNN
        x_cnn = x.permute(0, 2, 1) 
        x_cnn = self.cnn(x_cnn) 
        
        # Transformer
        x_trans = x_cnn.permute(0, 2, 1)
        x_trans = self.pos_encoder(x_trans)
        x_trans = self.transformer_encoder(x_trans)
        
        # Head
        x_flat = x_trans.contiguous().view(x_trans.size(0), -1)
        out = self.fc(x_flat)
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=2, type=int)
    parser.add_argument('-data', default='../data/mix.csv', type=str)
    
    args = parser.parse_args()
    model_path = 'model/hybrid_model.pth'

    test_data = generate_test_data(args.data, args.window_size)
    if len(test_data) == 0: exit()

    model = HybridModel(NUM_CLASSES, args.hidden_size, args.window_size).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Run hybrid_train.py first.")
        exit()
    
    model.eval()
    
    TP, FP, TN, FN = 0, 0, 0, 0
    latencies = []
    
    process = psutil.Process(os.getpid())
    process.cpu_percent() 
    
    print("Starting evaluation timer...")
    start_time = time.time()
    
    with torch.no_grad():
        for pattern, target_eid, label in test_data:
            pattern_tensor = torch.tensor([pattern], dtype=torch.long).to(device)
            
            t0 = time.time()
            output = model(pattern_tensor)
            latencies.append((time.time() - t0) * 1000)
            
            probs = torch.argsort(output, 1, descending=True)
            top_k = probs[0, :args.num_candidates].tolist()
            
            predicted_anomaly = 0 if target_eid in top_k else 1
            
            if label == 1:
                if predicted_anomaly == 1: TP += 1
                else: FN += 1
            else:
                if predicted_anomaly == 1: FP += 1
                else: TN += 1

    total_time = time.time() - start_time
    end_ram = process.memory_info().rss / 1024 / 1024
    cpu_usage = process.cpu_percent()
    
    P = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    R = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    FPR = 100 * FP / (FP + TN) if (FP + TN) > 0 else 0
    Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)

    print('=' * 40)
    print(f"  EVALUATION REPORT - CNN+TRANSFORMER HYBRID")
    print('=' * 40)
    print(f"CONFUSION MATRIX:")
    print(f"  TP: {TP:<5} | FP: {FP:<5}")
    print(f"  FN: {FN:<5} | TN: {TN:<5}")
    print('-' * 40)
    print(f"ACCURACY METRICS:")
    print(f"Accuracy:      {Accuracy:.3f}%")
    print(f'-' * 5)
    print(f"Precision:     {P:.3f}%")
    print(f"Recall:        {R:.3f}%")
    print(f"F1 Score:      {F1:.3f}%")
    print(f"FPR:           {FPR:.3f}%")
    print('-' * 40)
    print(f"PERFORMANCE METRICS:")
    print(f"Total Time:    {total_time:.2f}s")
    print(f"Throughput:    {len(test_data)/total_time:.0f} logs/sec")
    print(f"Avg Latency:   {np.mean(latencies):.4f} ms")
    print(f"P99 Latency:   {np.percentile(latencies, 99):.4f} ms")
    print(f"CPU Usage:     {cpu_usage:.1f}%")
    print(f"RAM Usage:     {end_ram:.1f} MB")
    print('=' * 40)