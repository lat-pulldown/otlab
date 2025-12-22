import time
import requests
import json
import re
import subprocess
import os

# --- CONFIGURATION ---
LOG_FILE = '/home/ubuntu/conpot/logs/conpot.log'
THINGSBOARD_HOST = 'http://<VM_IP>:8080' # <--- YOUR VM IP
ACCESS_TOKEN = 'DEVICE_ACCESS_TOKEN'  # <--- YOUR TOKEN
URL = f'{THINGSBOARD_HOST}/api/v1/{ACCESS_TOKEN}/telemetry'

# --- REGEX PATTERNS ---
# Matches: "New Modbus connection from 172.19.0.1:52324"
REGEX_CONN = r"New (.*?) connection from ([\d\.]+):(\d+)"
# Matches: "Function 43 can not be broadcasted"
REGEX_SCAN = r"Function (\d+) can not be broadcasted"
# Matches: "Modbus traffic from ... {'request': ...}"
REGEX_PAYLOAD = r"Modbus traffic from .*?request': b'([a-fA-F0-9]+)'"

def send_telemetry(data):
    try:
        requests.post(URL, json=data)
        print(f"[>] Sent: {data}")
    except Exception as e:
        print(f"[!] Error: {e}")

def tail_f(filename):
    # Standard tail -F implementation
    p = subprocess.Popen(['tail', '-F', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        line = p.stdout.readline()
        if line:
            yield line.decode('utf-8').strip()
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    print("--- Conpot Log Shipper Started ---")
    
    # 1. Send a "Hello" packet so ThingsBoard learns the keys immediately
    send_telemetry({"last_log": "Script Started", "attack_intensity": 0, "connection_count": 0})

    try:
        for line in tail_f(LOG_FILE):
            
            # SCENARIO 1: General Connection (The "Log")
            conn_match = re.search(REGEX_CONN, line)
            if conn_match:
                protocol = conn_match.group(1) # e.g. Modbus
                ip = conn_match.group(2)       # e.g. 172.19.0.1
                
                # Send telemetry for the "Connection Log"
                payload = {
                    "last_log": f"Connection from {ip} ({protocol})",
                    "source_ip": ip,
                    "connection_count": 1  # We can sum this in graphs
                }
                send_telemetry(payload)
                continue

            # SCENARIO 2: Suspicious Scan (The "Alert")
            scan_match = re.search(REGEX_SCAN, line)
            if scan_match:
                func = scan_match.group(1)
                payload = {
                    "last_log": f"Suspicious Scan: Function {func}",
                    "attack_intensity": 5, # Red line spikes to 5
                    "alert_status": "WARNING"
                }
                send_telemetry(payload)
                continue

            # SCENARIO 3: Malicious Payload (The "Critical Alert")
            if "request': b'" in line:
                payload = {
                    "last_log": "Malicious Payload Detected",
                    "attack_intensity": 10, # Red line spikes to 10
                    "alert_status": "CRITICAL"
                }
                send_telemetry(payload)

    except KeyboardInterrupt:
        print("Stopped.")

