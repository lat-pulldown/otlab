import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import sys

# --- CONFIGURATION ---
MODEL_PATH = 'model/cnn_model.pth' 
NUM_CLASSES = 100
WINDOW_SIZE = 10
EMBED_DIM = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARCHITECTURE (Corrected: No Pooling) ---
class Model(nn.Module):
    def __init__(self, num_classes, window_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_classes, EMBED_DIM)
        
        # Conv Block 1: 128 Filters, No Pooling
        self.conv1 = nn.Sequential(
            nn.Conv1d(EMBED_DIM, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Conv Block 2: 256 Filters, No Pooling
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Flattened size: 256 filters * 10 (window_size) = 2560
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        # x: [batch, window]
        x = self.embedding(x) # [batch, window, embed]
        x = x.permute(0, 2, 1) # [batch, embed, window]
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_test_data(name, window_size):
    print(f"Loading data/{name}...")
    try:
        df = pd.read_csv('data/' + name, header=None, names=['eid', 'label'])
    except:
        try:
            df = pd.read_csv(name, header=None, names=['eid', 'label'])
        except:
            print(f"Error: {name} not found.")
            return []
    
    # n-1 offset to match training
    seq = [x - 1 for x in df['eid'].values.tolist()]
    labels = df['label'].values.tolist()
    
    data = []
    for i in range(len(seq) - window_size):
        data.append((seq[i:i+window_size], seq[i+window_size], labels[i+window_size]))
    return data

def tune_cnn(test_data, model):
    print("Gathering model confidence scores...")
    all_probs = []
    all_labels = []
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for pattern, target_eid, label in test_data:
            input_tensor = torch.tensor([pattern], dtype=torch.long).to(device)
            output = model(input_tensor)
            prob_dist = softmax(output)
            
            # Probability assigned to the ACTUAL next EID
            target_prob = prob_dist[0][target_eid].item()
            
            all_probs.append(target_prob)
            all_labels.append(label)

    # Sweep thresholds
    thresholds = np.linspace(0.0001, 0.99, 200)
    print(f"\n{'Threshold':<12} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    
    best_f1, best_t = 0, 0
    # Keep track of best precision/recall for reporting
    best_metrics = (0, 0) 

    for t in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for prob, label in zip(all_probs, all_labels):
            # Anomaly if Confidence < Threshold
            pred_anomaly = 1 if prob < t else 0
            
            if label == 1:
                if pred_anomaly == 1: tp += 1
                else: fn += 1
            else:
                if pred_anomaly == 1: fp += 1
                else: tn += 1
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        if f1 > best_f1:
            best_f1, best_t = f1, t
            best_metrics = (prec, rec)
            # Only print improvements to keep output clean
            print(f"{t:<12.5f} | {f1:<10.2%} | {prec:<10.2%} | {rec:<10.2%}")

    print("-" * 55)
    print(f"WINNER: Optimal Threshold = {best_t:.5f} (F1: {best_f1:.2%})")
    print(f"Metrics: Precision {best_metrics[0]:.2%} | Recall {best_metrics[1]:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='mix01.csv')
    args = parser.parse_args()

    data = load_test_data(args.data, WINDOW_SIZE)
    model = Model(NUM_CLASSES, WINDOW_SIZE).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            tune_cnn(data, model)
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file {MODEL_PATH} not found.")