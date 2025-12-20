import pandas as pd
import re
import os
import json
import argparse
import numpy as np

# ==========================================
#      CONFIGURATION & RULES
# ==========================================
# 1. IP WHITELIST
NORMAL_IP = "YOUR CALDERA IP", "YOUR LOCAL IP"

# 2. TEMPLATE WHITELIST (STRICT)
# Only these specific actions from NORMAL_IP are considered Safe (Label 0).
# Everything else (Scans, HTTP, Timeouts, Errors) will be Label 1.
NORMAL_TEMPLATES = [
    # A. The Startup Sequence (Lines 1-43)
    "Startup_Init",           # Covers "Starting Conpot", "Found and enabled", "Serving TCP/IP"
    
    # B. The Normal Polling Loop (Steady State)
    "Data Transfer",          # The main "Modbus traffic from..."
    "Response Sent"           # The corresponding "Modbus response sent to..."
]

class LogParser:
    def __init__(self, eid_map_file='eid_mapping.json'):
        self.eid_map_file = eid_map_file
        self.eid_map = self.load_eid_map()
        self.next_eid = max(self.eid_map.values()) + 1 if self.eid_map else 1
        
        # 3. PATTERN DEFINITIONS
        # These define "What happened". 
        # "Is it safe?" is determined ONLY by the whitelist above.
        self.patterns = [
            # --- SYSTEM / STARTUP (Whitelisted) ---
            (r"Starting Conpot", "Startup_Init"), 
            (r"Found and enabled", "Startup_Init"),
            (r"Serving TCP/IP", "Startup_Init"),
            (r"Initializing Virtual File System", "Startup_Init"),
            (r"Fetched .* as external ip", "Startup_Init"),
            (r"Class .* Instance .* Attribute", "Startup_Init"),
            (r"Creating persistent data store", "Startup_Init"),
            (r"FTP Serving File System", "Startup_Init"),
            (r"TFTP Serving File System", "Startup_Init"),
            (r"server started on", "Startup_Init"),
            (r"handle server PID", "Startup_Init"),
            (r"Using default FS path", "Startup_Init"),
            (r"--force option specified", "Startup_Init"),

            # --- NORMAL POLLING (Whitelisted) ---
            (r"Modbus traffic from ([\d\.]+)", "Data Transfer"),
            (r"Modbus response sent to ([\d\.]+)", "Response Sent"),

            # --- ATTACKS & ANOMALIES (NOT Whitelisted -> Label 1) ---
            (r"Exception occurred", "Server Error"),
            (r"Traceback", "Server Crash"),
            (r"New modbus session from ([\d\.]+)", "Scan_Session"),    # Scanning
            (r"New Modbus connection from ([\d\.]+)", "Scan_Connection"),# Scanning
            (r"New http session from ([\d\.]+)", "HTTP_Scan"),          # Caldera / Scan
            (r"HTTP/.* request from", "HTTP_Request"),                  # Caldera Attack
            (r"Function 3 can not be broadcasted", "Broadcast Error"),
            (r"Modbus connection terminated", "Connection Terminated"), # Usually scan ending
            (r"Session timed out", "Timeout")
        ]

    def load_eid_map(self):
        if os.path.exists(self.eid_map_file):
            with open(self.eid_map_file, 'r') as f:
                return json.load(f)
        return {}

    def save_eid_map(self):
        with open(self.eid_map_file, 'w') as f:
            json.dump(self.eid_map, f, indent=4)

    def get_eid(self, template):
        if template not in self.eid_map:
            self.eid_map[template] = self.next_eid
            self.next_eid += 1
        return self.eid_map[template]

    def parse(self, log_path):
        print(f"[-] Parsing Log: {log_path}")
        data = []
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                ts_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                if not ts_match: continue
                timestamp_str = ts_match.group(1)
                content = line[len(timestamp_str):].strip()
                
                template = "Unknown"
                source_ip = "None"
                
                for pattern, templ_name in self.patterns:
                    match = re.search(pattern, content)
                    if match:
                        template = templ_name
                        if match.groups():
                            for g in match.groups():
                                if g and re.match(r"^\d{1,3}\.", g):
                                    source_ip = g
                                    break
                        break
                
                if source_ip == "None":
                    ip_search = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", content)
                    if ip_search: source_ip = ip_search.group(1)

                # --- LABELING LOGIC (STRICT) ---
                label = 1 # Default to Attack / Abnormal
                
                # Only change to 0 if explicitly safe
                if (source_ip == NORMAL_IP) and (template in NORMAL_TEMPLATES):
                    label = 0
                elif template in ["Startup_Init", "Server Error"]:
                    # Note: You can remove "Server Error" from here if you want errors to be Label 1
                    label = 0
                elif template == "Broadcast Error":
                    label = 0

                eid = self.get_eid(template)
                
                data.append({
                    'timestamp': pd.to_datetime(timestamp_str.replace(',', '.')),
                    'eid': eid,
                    'label': label
                })
                
        self.save_eid_map()
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df

def generate_deeplog_csv(df, output_path, mode):
    print(f"[-] Generating DeepLog CSV ({mode}): {output_path}")
    if mode == 'train':
        df_clean = df[df['label'] == 0]
        df_clean[['eid']].to_csv(output_path, index=False, header=False)
    else:
        df[['eid', 'label']].to_csv(output_path, index=False, header=False)

def generate_transformer_csv(df, temp_csv_path, output_base, mode):
    print(f"[-] Generating Transformer CSVs ({mode})...")
    
    if mode == 'train':
        df = df[df['label'] == 0]
    
    df_window = df.resample('5s').agg({
        'eid': ['count', 'nunique'],
        'label': 'max'
    })
    df_window.columns = ['eid_count', 'eid_nunique', 'label']
    df_window['label'] = df_window['label'].fillna(0).astype(int)
    
    num_windows = len(df_window)
    
    if os.path.exists(temp_csv_path):
        print(f"    Merging temperature from {temp_csv_path} (Sequential Mode)...")
        try:
            df_temp = pd.read_csv(temp_csv_path)
            temp_col = [c for c in df_temp.columns if 'Temp' in c or 'temp' in c]
            if not temp_col:
                raise ValueError("No 'Temp' column found in CSV")
            temp_col = temp_col[0]
            temp_values = df_temp[temp_col].values
            
            if len(temp_values) > 0:
                if len(temp_values) < num_windows:
                    repeats = (num_windows // len(temp_values)) + 1
                    temp_values = np.tile(temp_values, repeats)
                temp_values = temp_values[:num_windows]
                df_window['temperature'] = temp_values
            else:
                print("    [!] Warning: Temperature file is empty. Using 0.0")
                df_window['temperature'] = 0.0
        except Exception as e:
            print(f"    [!] Error reading temp file: {e}")
            df_window['temperature'] = 0.0
    else:
        print(f"    [!] Warning: {temp_csv_path} not found. Filling temperature with 0.")
        df_window['temperature'] = 0.0

    df_window['eid_count'] = df_window['eid_count'].fillna(0)
    df_window['eid_nunique'] = df_window['eid_nunique'].fillna(0)
    
    df_window.reset_index(inplace=True)
    df_window.rename(columns={'index': 'timestamp'}, inplace=True)
    
    out_path_tf = f"{output_base}_tf.csv"
    cols_temp = ['timestamp', 'eid_count', 'eid_nunique', 'temperature', 'label']
    df_window[cols_temp].to_csv(out_path_tf, index=False)
    print(f"    Saved ({mode}): {out_path_tf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'predict'], required=True)
    parser.add_argument("--log", required=True, help="Input log file", default="mix.log")
    parser.add_argument("--temp", default="temp.csv", help="Temperature CSV file")
    parser.add_argument("--out", default="mix.csv", help="Output for DeepLog")
    parser.add_argument("--out_tf", default="mix", help="Base name for Transformer outputs")
    
    args = parser.parse_args()

    if args.mode == 'train':
        args.out_dl = "train.csv"
        args.out_tf_base = "train_tf.csv"

    log_processor = LogParser()
    df_logs = log_processor.parse(args.log)
    
    if df_logs.empty:
        print("Error: Log file is empty or could not be parsed.")
        return

    generate_deeplog_csv(df_logs, args.out_dl, args.mode)
    generate_transformer_csv(df_logs, args.temp, args.out_tf_base, args.mode)
    print("[+] Processing Complete.")

if __name__ == "__main__":
    main()