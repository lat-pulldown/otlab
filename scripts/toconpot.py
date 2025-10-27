### Conpot to Thingsboard

import time
import requests

# ThingsBoard device token
DEVICE_TOKEN = "IEw4RCLVVIUTFoC2F9bj"
TB_URL = f"http://localhost:8080/api/v1/{DEVICE_TOKEN}/telemetry"

# Path to Conpot log
LOG_FILE = "/Users/renshiro/conpot/data/conpot.log"

def parse_line(line):
    """
    Simple parser: convert a plain text log line to JSON.
    Adjust this based on your log format.
    """
    data = {}
    # Example: "[2025-10-26 12:00:00] INFO: Modbus read from 127.0.0.1"
    try:
        timestamp = line.split("]")[0].strip("[")
        data["timestamp"] = timestamp
        if "Modbus read" in line:
            data["event"] = "modbus_read"
        elif "Modbus write" in line:
            data["event"] = "modbus_write"
        else:
            data["event"] = "other"
    except:
        data["event"] = "parse_error"
    return data

def send_to_tb(data):
    try:
        requests.post(TB_URL, json=data)
    except Exception as e:
        print("Failed to send:", e)

def tail_f(file):
    """
    Like 'tail -f': keep reading new lines
    """
    with open(file, "r") as f:
        # Go to end of file
        f.seek(0,2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            yield line

if __name__ == "__main__":
    for line in tail_f(LOG_FILE):
        data = parse_line(line)
        send_to_tb(data)