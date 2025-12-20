import torch
import torch.nn as nn
import time
import argparse
import pandas as pd
import os
import numpy as np
import psutil

# --- CONFIGURATION ---
MODEL_PATH = 'model/cnn_model.pth' 
NUM_CLASSES = 100
WINDOW_SIZE = 10
EMBED_DIM = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARCHITECTURE (Matched to your trained model) ---
class Model(nn.Module):
    def __init__(self, num_classes, window_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_classes, EMBED_DIM)
        
        # Conv Block 1: 128 Filters
        self.conv1 = nn.Sequential(
            nn.Conv1d(EMBED_DIM, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Conv Block 2: 256 Filters
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Flattened size: 256 filters * 10 window_size = 2560
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        x = self.embedding(x)       # [batch, window, embed]
        x = x.permute(0, 2, 1)      # [batch, embed, window]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.fc(x)
        return x

def load_test_data(name, window_size):
    print(f"Loading {name}...")
    try:
        # Check local folder first, then data/ folder
        if os.path.exists(name):
            path = name
        elif os.path.exists('data/' + name):
            path = 'data/' + name
        else:
            raise FileNotFoundError
            
        df = pd.read_csv(path, header=None, names=['eid', 'label'])
    except:
        print(f"Error: File {name} not found.")
        return []
    
    # Apply n-1 offset to match training logic
    seq = [x - 1 for x in df['eid'].values.tolist()]
    labels = df['label'].values.tolist()
    
    data = []
    for i in range(len(seq) - window_size):
        data.append((seq[i:i+window_size], seq[i+window_size], labels[i+window_size]))
    print(f"Total test windows: {len(data)}")
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Defaults to mix01.csv and your optimal threshold
    parser.add_argument('-data', default='mix01.csv')
    parser.add_argument('-threshold', default=0.00010, type=float)
    args = parser.parse_args()

    test_data = load_test_data(args.data, WINDOW_SIZE)
    if not test_data: exit()

    model = Model(NUM_CLASSES, WINDOW_SIZE).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model.eval()

    # --- PERFORMANCE METRICS SETUP ---
    latencies = []
    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss / 1024 / 1024 # MB
    cpu_start = process.cpu_percent()
    
    print(f"Starting evaluation with Threshold < {args.threshold:.5f} ...")
    
    TP, FP, TN, FN = 0, 0, 0, 0
    softmax = nn.Softmax(dim=1)
    
    start_time = time.time()
    
    with torch.no_grad():
        for pattern, target_eid, truth_label in test_data:
            t0 = time.time()
            
            # Prepare Input
            input_tensor = torch.tensor([pattern], dtype=torch.long).to(device)
            
            # Forward Pass
            output = model(input_tensor)
            prob_dist = softmax(output)
            
            # Get probability of the ACTUAL event that occurred
            confidence = prob_dist[0][target_eid].item()
            
            # ANOMALY LOGIC: 
            # If the model assigns very low probability (< threshold) to what actually happened,
            # it means this event was "unexpected" -> Anomaly.
            is_anomaly = 1 if confidence < args.threshold else 0
            
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # ms

            if is_anomaly:
                if truth_label == 1: TP += 1
                else: FP += 1
            else:
                if truth_label == 0: TN += 1
                else: FN += 1

    total_time = time.time() - start_time
    
    # --- RESOURCE & TIMING METRICS ---
    end_ram = process.memory_info().rss / 1024 / 1024 # MB
    # CPU percent since last call
    cpu_usage = process.cpu_percent() 
    
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = len(test_data) / total_time
    
    # Accuracy Metrics
    P = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    R = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    FPR = 100 * FP / (FP + TN) if (FP + TN) > 0 else 0
    Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)

    # --- FULL EVALUATION REPORT ---
    print('=' * 40)
    print(f"  EVALUATION REPORT - 1D-CNN")
    print('=' * 40)
    print(f"CONFUSION MATRIX:")
    print(f"  TP: {TP:<5} | FP: {FP:<5}")
    print(f"  FN: {FN:<5} | TN: {TN:<5}")
    print('-' * 40)
    print(f"ACCURACY METRICS:")
    print(f"  Accuracy:  {Accuracy:.3f}%")
    print(f'-' * 5)
    print(f"  Precision: {P:.3f}%")
    print(f"  Recall:    {R:.3f}%")
    print(f"  F1-Score:  {F1:.3f}%")
    print(f"  FPR:       {FPR:.3f}%")
    print('-' * 40)
    print(f"PERFORMANCE METRICS:")
    print(f"  Total Time:   {total_time:.1f}s")
    print(f"  Throughput:   {throughput:.0f} logs/sec")
    print(f"  Avg Latency:  {avg_latency:.3f} ms")
    print(f"  P99 Latency:  {p99_latency:.3f} ms")
    print(f"  CPU Usage:    {cpu_usage:.1f}%")
    print(f"  RAM Usage:    {end_ram:.1f} MB")
    print('=' * 40)