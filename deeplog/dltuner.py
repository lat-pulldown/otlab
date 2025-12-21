import torch
import torch.nn as nn
import time
import argparse
import pandas as pd
import os
import sys
import numpy as np

# --- CONFIGURATION (Matches model_test.py) ---
MODEL_PATH = 'model/adam_batch_size=2048;epoch=50.pt'
NUM_CLASSES = 100
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (Strictly matching model_test.py) ---
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

def generate_test_data(name, window_size):
    print(f"Loading {name}...")
    try:
        # Assumes data is in the 'data/' folder per your model_test logic
        df = pd.read_csv('../data/' + name, header=None, names=['eid', 'label'])
    except FileNotFoundError:
        try:
            df = pd.read_csv(name, header=None, names=['eid', 'label'])
        except:
            print(f"Error: File {name} not found.")
            return []
    
    # CRITICAL: Match the n-1 indexing from model_test.py
    seq = [x - 1 for x in df['eid'].values.tolist()]
    labels = df['label'].values.tolist()
    
    data = []
    for i in range(len(seq) - window_size):
        pattern = seq[i:i + window_size]
        target_eid = seq[i + window_size]
        target_label = labels[i + window_size]
        data.append((pattern, target_eid, target_label))
    return data

def evaluate(model, test_data, k, window_size):
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for pattern, target_eid, label in test_data:
            # Prepare Input (float tensor + view)
            seq = torch.tensor(pattern, dtype=torch.float).view(-1, window_size, INPUT_SIZE).to(device)
            
            output = model(seq)
            
            # Get Top-K Predictions (Matching model_test.py logic)
            # torch.argsort(output, 1)[0] gives indices from lowest to highest score
            # [-k:] picks the k highest score indices
            predicted = torch.argsort(output, 1)[0][-k:].tolist()
            
            # Anomaly Prediction
            is_anomaly = target_eid not in predicted
            
            if is_anomaly:
                if label == 1: TP += 1
                else: FP += 1
            else:
                if label == 0: TN += 1
                else: FN += 1
    
    P = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    R = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return P, R, F1, FP, FN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='../data/mix.csv', type=str)
    parser.add_argument('-window_size', default=10, type=int) 
    args = parser.parse_args()

    test_data = generate_test_data(args.data, args.window_size)
    if not test_data: sys.exit(1)

    print("Loading Model...")
    model = Model(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"\n{'K':<5} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10} | {'FP':<6} | {'FN':<6}")
    print("-" * 65)

    # Sweep K from 1 to 15
    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    best_f1, best_k = 0, 0

    for k in candidates:
        p, r, f1, fp, fn = evaluate(model, test_data, k, args.window_size)
        print(f"{k:<5} | {f1:<10.2f} | {p:<10.2f} | {r:<10.2f} | {fp:<6} | {fn:<6}")
        
        if f1 > best_f1:
            best_f1, best_k = f1, k

    print("-" * 65)
    print(f"WINNER: Optimal K = {best_k} (F1: {best_f1:.2f}%)")
    print(f"Update your test command to: -num_candidates {best_k}")
    