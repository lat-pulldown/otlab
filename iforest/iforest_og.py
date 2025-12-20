import argparse
import pandas as pd
import numpy as np
import time
import os
import joblib
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
MODEL_PATH = 'model/iforest_model.pkl'
SCALER_PATH = 'model/iforest_scaler.pkl'
TRAIN_FILE = 'data/train.csv' 
DATA_DIR = 'data'

def load_data(filename, mode='train'):
    # Handle explicit paths or just filenames
    if os.path.exists(filename):
        path = filename
    else:
        path = os.path.join(DATA_DIR, filename)
    
    print(f"Loading {os.path.basename(path)}...")
    
    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        exit(1)

    # --- Data Parsing Logic ---
    # Matches your CNN/DeepLog logic: 
    # If 1 column -> Unlabeled Training Data
    # If 2 columns -> Labeled Test Data (Feature, Label)
    
    if mode == 'train':
        if df.shape[1] == 1:
            X = df.values
            y = None # Unsupervised
        else:
            # Fallback if train file has labels (ignore them)
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = None
    else: # test
        # Expecting [Feature, Label]
        if df.shape[1] >= 2:
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, 1].values
        else:
            # Fallback if test file has no labels
            X = df.values
            y = np.zeros(len(df)) # Dummy labels

    return X, y

def train(input_file):
    X, _ = load_data(input_file, mode='train')
    print(f"Total training samples: {len(X)}")

    # 1. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Model Init
    # contamination='auto' allows it to define its own threshold
    model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)

    print(f"Starting training (Isolation Forest)... Devices: CPU")
    start_time = time.time()

    # 3. Train
    model.fit(X_scaled)
    
    # Since iForest doesn't have epochs, we skip the epoch loop print
    # but strictly match the final output format.
    
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.3f}s")
    
    # 4. Save
    if not os.path.exists('model'):
        os.makedirs('model')
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")

def test(input_file):
    X, y_true = load_data(input_file, mode='test')
    print(f"Total test samples: {len(X)}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found. Run -mode train first.")
        return

    # Load Model & Scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded successfully.")

    # Prepare Resource Monitoring
    process = psutil.Process(os.getpid())
    process.cpu_percent() # Init call
    
    print("Starting evaluation timer...")
    
    # Scale Data beforehand (simulating preprocessing step)
    X_scaled = scaler.transform(X)

    # --- INFERENCE LOOP (To measure Latency/Throughput accurately) ---
    y_pred = []
    latencies = []
    start_eval_time = time.time()

    # We iterate to capture per-sample latency distribution (P99)
    # This simulates a real-time stream better than batch predict
    for i in range(len(X_scaled)):
        sample = X_scaled[i].reshape(1, -1)
        
        t0 = time.time()
        pred_raw = model.predict(sample) # Returns -1 (Anomaly) or 1 (Normal)
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000) # ms
        
        # Convert iForest output (-1/1) to Project Standard (1/0)
        # iForest: -1 is Anomaly. Project: 1 is Attack/Anomaly.
        is_anomaly = 1 if pred_raw[0] == -1 else 0
        y_pred.append(is_anomaly)

    total_time = time.time() - start_eval_time

    # --- METRICS ---
    # Resource Usage
    cpu_usage = process.cpu_percent()
    ram_usage = process.memory_info().rss / 1024 / 1024 # MB

    # Latency Stats
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = len(X) / total_time

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Derived Metrics
    accuracy = (tp + tn) / len(y_true) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

    # --- REPORT ---
    print('=' * 40)
    print(f"  EVALUATION REPORT - ISOLATION FOREST")
    print('=' * 40)
    print(f"CONFUSION MATRIX:")
    print(f"  TP: {tp:<5} | FP: {fp:<5}")
    print(f"  FN: {fn:<5} | TN: {tn:<5}")
    print('-' * 40)
    print(f"ACCURACY METRICS:")
    print(f"Accuracy:      {accuracy:.3f}%")
    print(f'-' * 5)
    print(f"Precision:     {precision:.3f}%")
    print(f"Recall:        {recall:.3f}%")
    print(f"F1 Score:      {f1:.3f}%")
    print(f"FPR:           {fpr:.3f}%")
    print('-' * 40)
    print(f"PERFORMANCE METRICS:")
    print(f"Total Time:    {total_time:.1f}s")
    print(f"Throughput:    {throughput:.0f} logs/sec")
    print(f"Avg Latency:   {avg_latency:.3f} ms")
    print(f"P99 Latency:   {p99_latency:.3f} ms")
    print(f"CPU Usage:     {cpu_usage:.1f}%")
    print(f"RAM Usage:     {ram_usage:.1f} MB")
    print('=' * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, required=True, choices=['train', 'test'])
    # Default to specific files if not provided, for convenience
    parser.add_argument('-data', type=str, default='mix.csv', help="Test data file")
    
    args = parser.parse_args()

    if args.mode == 'train':
        # Default train file is fixed in config, or could be arg
        train(TRAIN_FILE)
    elif args.mode == 'test':
        test(args.data)