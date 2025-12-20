import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import argparse
import os

FEATURE_COLS = ['eid_count', 'eid_nunique', 'temperature']

def tune(train_file, test_file):
    print(f"Loading Training: {train_file}...")
    print(f"Loading Testing: {test_file}...")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Isolation Forest...")
    model = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled)

    print("Scoring Test Data...")
    scores = model.score_samples(X_test_scaled)

    min_score, max_score = scores.min(), scores.max()
    thresholds = np.linspace(min_score, max_score, 200)

    best_f1, best_thresh = 0, 0

    for thresh in thresholds:
        y_pred = (scores < thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
            print(f"Threshold: {thresh:.5f} | F1: {f1:.2%}")

    print(f"\nWINNER: Best F1 Score: {best_f1:.2%}")
    print(f"Optimal Threshold: {best_thresh:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', default='data/train_tf01.csv')
    parser.add_argument('-data', default='data/mix_tf01.csv')
    args = parser.parse_args()
    tune(args.train, args.data)