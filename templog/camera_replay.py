import csv
import time
import requests
import sys
import os

# --- CONFIGURATION ---
# Make sure this filename matches exactly what is in your folder
CSV_FILE = 'temp.csv'

# ThingsBoard Settings
THINGSBOARD_HOST = 'http://<YOUR VM IP>:8080'  # Change IP if TB is on the VM
ACCESS_TOKEN = 'PASTE YOUR ACCESS TOKEN FROM THINGSBOARD'  # <--- Don't forget to paste the token from Step 1!
URL = f'{THINGSBOARD_HOST}/api/v1/{ACCESS_TOKEN}/telemetry'

# REPLAY SPEED
# 60 = Real-time (Wait 60 seconds between data points, just like the real camera)
# 1  = Fast Forward (Wait 1 second between points - BEST FOR TESTING)
SPEED_DELAY = 1 

def send_telemetry(temp_val, alarm_status, row_time):
    # Prepare the JSON payload
    payload = {
        "temperature": float(temp_val),
        "camera_status": alarm_status,
        "original_csv_time": row_time  # Useful to know which row we are on
    }
    
    try:
        # Send POST request to ThingsBoard
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            print(f"[SUCCESS] {row_time} -> Temp: {temp_val}°C | Status: {alarm_status}")
        else:
            print(f"[FAIL] Status Code: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"[ERROR] Could not connect to ThingsBoard: {e}")

if __name__ == "__main__":
    print(f"--- Starting Replay of {CSV_FILE} ---")
    print(f"Target: {URL}")
    print(f"Delay: {SPEED_DELAY} seconds between readings\n")

    if not os.path.exists(CSV_FILE):
        print(f"ERROR: File '{CSV_FILE}' not found.")
        print("Make sure the .py file and the .csv file are in the same folder.")
        sys.exit(1)

    try:
        with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
            # encoding='utf-8-sig' handles the BOM if Excel added it
            reader = csv.DictReader(f)
            
            # Verify columns exist before starting
            # We look for the exact headers from your file
            temp_col = 'Temp'
            status_col = 'Time'
            time_col = 'Time'

            if temp_col not in reader.fieldnames:
                print(f"ERROR: Could not find column '{temp_col}'")
                print(f"Found headers: {reader.fieldnames}")
                sys.exit(1)

            for row in reader:
                # Extract data using the specific OMRON headers
                temp = row[temp_col]
                status = row[status_col]
                timestamp = row[time_col]

                # Send it
                send_telemetry(temp, status, timestamp)

                # Wait
                time.sleep(SPEED_DELAY)

    except KeyboardInterrupt:
        print("\n--- Replay Stopped by User ---")