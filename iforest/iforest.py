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
# Isolation Forest MUST use the _tf (feature) dataset, not the _dl (sequence) dataset
TRAIN_FILE = 'data/train_tf01.csv' 
DATA_DIR = 'data'

# Define the numerical features we want to learn from
FEATURE_COLS = ['eid_count', 'eid_nunique', 'temperature']

def load_data(filename, mode='train'):
    if os.path.exists(filename):
        path = filename
    else:
        path = os.path.join(DATA_DIR, filename)
    
    print(f"Loading {os.path.basename(path)}...")
    
    try:
        # Read with header because _tf files have headers
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        exit(1)

    # 1. Select only the Numerical Features for X
    # We drop 'timestamp' (string) and 'label' (target)
    try:
        X = df[FEATURE_COLS].values
    except KeyError as e:
        print(f"Error: Dataset missing required columns {FEATURE_COLS}.")
        print(f"Are you using the correct '_tf.csv' file? Found columns: {df.columns.tolist()}")
        exit(1)

    # 2. Extract Labels (y) if they exist
    if 'label' in df.columns:
        y = df['label'].values
    else:
        # If unlabeled, create dummy zeros
        y = np.zeros(len(df))

    return X, y

def train(input_file):
    X, _ = load_data(input_file, mode='train')
    print(f"Total training samples: {len(X)}")

    # 1. Scaling (Crucial for iForest when mixing Counts and Temperature)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Model Init
    model = IsolationForest(n_estimators=100, contamination=0.001, n_jobs=-1, random_state=42)

    print(f"Starting training (Isolation Forest)... Devices: CPU")
    start_time = time.time()

    # 3. Train
    model.fit(X_scaled)
    
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

    process = psutil.Process(os.getpid())
    print("Starting evaluation timer...")
    
    # Pre-scale data
    X_scaled = scaler.transform(X)

    start_eval_time = time.time()
    latencies = []
    y_pred = []

    # Inference Loop
    for i in range(len(X_scaled)):
        sample = X_scaled[i].reshape(1, -1)
        
        t0 = time.time()
        score = model.score_samples(sample)
        # pred_raw = model.predict(sample) 
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000)
        
        # Convert iForest output (-1 is Anomaly, 1 is Normal) to (1 is Attack, 0 is Normal)
        # Change the threshhold when using custom dataset. Use iftuner.py to find threshhold. Default set to -0.51283
        is_anomaly = 1 if score[0] < -0.51283 else 0
        # is_anomaly = 1 if pred_raw[0] == -1 else 0
        y_pred.append(is_anomaly)

    total_time = time.time() - start_eval_time

    # Metrics
    cpu_usage = process.cpu_percent()
    ram_usage = process.memory_info().rss / 1024 / 1024
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = len(X) / total_time

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / len(y_true) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

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
    parser.add_argument('-data', type=str, default='data/mix_tf01.csv', help="Data file")
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(TRAIN_FILE)
    elif args.mode == 'test':
        test(args.data)