# v2

import torch
import torch.nn as nn
import time
import argparse
import pandas as pd
import psutil
import os
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *** CONFIGURATION ***
NUM_CLASSES = 100 
PREDICT_FILE = 'mix.csv'

def generate_test_data(name):
    print(f"Loading {name}...")
    try:
        df = pd.read_csv('../data/' + name, header=None, names=['eid', 'label'])
    except FileNotFoundError:
        print(f"Error: File data/{name} not found.")
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

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    # Hyperparameters
    num_classes = NUM_CLASSES
    input_size = 1
    model_path = 'model/adam_batch_size=2048;epoch=50.pt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=2, type=int) 
    # NEW: Add the data argument
    parser.add_argument('-data', default='../data/mix.csv', type=str, help='Filename of the test data inside data/ folder')
    
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates
    predict_file_name = args.data  # NEW: Get filename from arguments

    # Load Model
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Run train first.")
        exit()
    model.eval()

    # NEW: Pass the argument variable instead of the hardcoded constant
    test_data = generate_test_data(predict_file_name) 
    if not test_data: exit()

    TP, FP, TN, FN = 0, 0, 0, 0
    
    
    # --- PERFORMANCE METRICS SETUP ---
    latencies = []
    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss / 1024 / 1024 # MB
    cpu_start = process.cpu_percent()
    
    print(f"Starting evluation timer... (Initial RAM: {start_ram:.2f} MB)")
    start_time = time.time()
    
    with torch.no_grad():
        for pattern, target_eid, truth_label in test_data:
            
            # Measure Latency per inference
            t0 = time.time()
            
            # Prepare Input
            seq = torch.tensor(pattern, dtype=torch.float).view(-1, window_size, input_size).to(device)
            
            # Forward Pass
            output = model(seq)
            
            # Get Top-K Predictions
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            
            # Latency End
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # Convert to ms
            
            # ANOMALY DETECTION LOGIC
            model_is_anomaly = target_eid not in predicted
            
            if model_is_anomaly:
                if truth_label == 1: TP += 1
                else: FP += 1
            else:
                if truth_label == 0: TN += 1
                else: FN += 1

    total_time = time.time() - start_time
    
    # --- RESOURCE METRICS END ---
    end_ram = process.memory_info().rss / 1024 / 1024 # MB
    peak_ram = end_ram # Approximation
    cpu_usage = process.cpu_percent() # Average CPU usage since last call
    
    # --- CALCULATIONS ---
    # Accuracy Metrics
    P = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    R = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    FPR = 100 * FP / (FP + TN) if (FP + TN) > 0 else 0
    Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
    
    # Performance Metrics
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99) # 99th percentile (Worst case)
    throughput = len(test_data) / total_time
    
    print('=' * 40)
    print(f"  EVALUATION REPORT - DEEPLOG")
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