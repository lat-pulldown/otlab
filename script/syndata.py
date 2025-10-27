#!/usr/bin/env python3
"""
syndata.py

- Generates continuous synthetic hybrid logs (protocol events + temperature readings)
  and sends them to ThingsBoard as time-series telemetry (ts + values).
- Optionally tails a real conpot.log and forwards parsed real events to ThingsBoard.
- Default behavior: run until RUN_DURATION_MIN minutes elapse, or forever if set to 0/None.

Edit the CONFIG block below before running.
"""

import time
import json
import random
import threading
import requests
import re
from datetime import datetime

# ----------------- CONFIG -----------------
DEVICE_TOKEN = "IEw4RCLVVIUTFoC2F9bj"         # <--- set this
TB_HOST = "http://localhost:8080"          # ThingsBoard base URL (change if remote)
TB_URL = f"{TB_HOST}/api/v1/{DEVICE_TOKEN}/telemetry"
LOG_FILE = "/Users/renshiro/conpot/data/conpot.log"           # <--- set this to your conpot.log path
GENERATE_SYNTHETIC = True                  # generate synthetic hybrid logs
FORWARD_CONPOT_LOGS = True                 # tail and forward real conpot.log
SYNTHETIC_INTERVAL_SEC = 1.0               # how often synthetic logs are sent (sec)
RUN_DURATION_MIN = 60                      # run time in minutes; 0 or None => run forever
IGNORE_STARTUP_KEYWORDS = True             # drop Conpot startup noise lines (recommended)
# ------------------------------------------

# regex for parsing Conpot "Class ..." lines and extracting class/instance/attribute
CLASS_RE = re.compile(r"Class\s+(\d+)[^,]*,\s*Instance\s+(\d+),\s*Attribute\s+(\d+)", re.IGNORECASE)
IP_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")

# Helper: POST to ThingsBoard
def send_payload_to_tb(payload):
    try:
        r = requests.post(TB_URL, json=payload, timeout=5)
        if r.status_code >= 400:
            print(f"[WARN] ThingsBoard returned {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[ERROR] Failed to post to ThingsBoard: {e}")

# Synthetic hybrid generator
def synthetic_generator(stop_event):
    protocol_list = ["bacnet", "modbus", "enip", "ftp"]
    event_list = ["read", "write", "connect", "disconnect"]
    node_prefix = "10.10.10."

    # thermal baseline parameters
    base_temp = 30.0   # baseline temperature (deg C)
    base_noise = 0.4   # small random noise
    spike_prob = 0.005 # probability per sample to generate a spike (simulate anomaly)
    spike_magnitude = 15.0

    print("[SYN] Synthetic generator started.")
    while not stop_event.is_set():
        now_ms = int(time.time() * 1000)

        # Create synthetic protocol event
        protocol = random.choice(protocol_list)
        event = random.choices(event_list, weights=[0.6, 0.1, 0.25, 0.05])[0]  # more reads/connects

        # fake source IP (internal network style)
        src_ip = node_prefix + str(random.randint(2, 200))

        # generate class/instance/attribute for bacnet-like logs
        clazz = random.randint(1, 30)
        instance = random.randint(1, 6)
        attribute = random.randint(1, 8)

        # temperature reading (thermal camera mock)
        temp = base_temp + random.gauss(0, base_noise)
        if random.random() < spike_prob:
            # occasional spike representing a thermal anomaly
            temp += spike_magnitude * (0.5 + random.random())

        # Compose the telemetry payload (time-series form)
        values = {
            "source": "synthetic",
            "protocol": protocol,
            "event": event,
            "source_ip": src_ip,
            "class": clazz,
            "instance": instance,
            "attribute": attribute,
            "temperature": round(temp, 2),
            "details": f"synthetic {protocol} {event} from {src_ip}"
        }

        # label synthetic stream as normal by default
        values["label"] = "normal"

        payload = {"ts": now_ms, "values": values}
        send_payload_to_tb(payload)

        # also send a thermal-only telemetry record
        therm_payload = {
            "ts": now_ms,
            "values": {
                "source": "thermal_sim",
                "temperature": round(temp, 2),
                # mark anomaly if temp is significantly above baseline
                "label": "anomaly" if temp > (base_temp + spike_magnitude * 0.4) else "normal",
            }
        }
        send_payload_to_tb(therm_payload)

        # sleep
        for _ in range(int(max(1, SYNTHETIC_INTERVAL_SEC * 10))):
            if stop_event.is_set():
                break
            time.sleep(SYNTHETIC_INTERVAL_SEC / 10.0)

    print("[SYN] Synthetic generator stopped.")

# Parse conpot log line into structured dict (returns None if ignored)
def parse_conpot_line(line):
    line = line.strip()
    if not line:
        return None

    # Optionally ignore startup noise
    if IGNORE_STARTUP_KEYWORDS:
        startup_keywords = [
            "Please wait while the system copies",
            "Starting Conpot using configuration",
            "Using default FS path",
            "Initializing Virtual File System",
            "Fetched",
            "Found and enabled",
            "Creating persistent data store"
        ]
        for kw in startup_keywords:
            if kw.lower() in line.lower():
                return None

    # try to parse timestamp at the start: "YYYY-MM-DD HH:MM:SS,ms"
    ts_ms = int(time.time() * 1000)
    try:
        # take first two tokens as timestamp (date + time)
        parts = line.split(" ", 2)
        if len(parts) >= 2:
            ts_str = parts[0] + " " + parts[1]
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
            ts_ms = int(dt.timestamp() * 1000)
    except Exception:
        # fallback: use current time
        pass

    # find any IP in the line
    src_ip = None
    ipm = IP_RE.search(line)
    if ipm:
        src_ip = ipm.group(1)

    # Look for Class/Instance/Attribute
    m = CLASS_RE.search(line)
    if m:
        clazz = int(m.group(1))
        instance = int(m.group(2))
        attribute = int(m.group(3))
        event_type = "class_read"
    else:
        clazz = instance = attribute = None
        # Heuristic event detection
        low = line.lower()
        if "server started" in low or "started on" in low:
            event_type = "server_started"
        elif "handle server pid" in low:
            event_type = "server_pid"
        elif "responding to external" in low or "responding" in low:
            event_type = "connection_event"
        else:
            event_type = "log"

    structured = {
        "ts": ts_ms,
        "values": {
            "source": "conpot",
            "event": event_type,
            "details": line,
            "source_ip": src_ip,
            "protocol": None,
            "class": clazz,
            "instance": instance,
            "attribute": attribute,
        }
    }

    # detect protocol names in the line
    for proto in ("bacnet", "enip", "ftp", "modbus", "s7"):
        if proto in line.lower():
            structured["values"]["protocol"] = proto
            break

    # do not auto-label forwarded conpot logs (we'll label later)
    structured["values"]["label"] = "unknown"

    return structured

# Tail conpot.log and forward parsed lines
def tail_conpot_forwarder(stop_event, logfile_path):
    print(f"[TAIL] Starting tail of {logfile_path}")
    try:
        with open(logfile_path, "r") as f:
            f.seek(0, 2)  # move to EOF
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                parsed = parse_conpot_line(line)
                if parsed:
                    send_payload_to_tb(parsed)
    except FileNotFoundError:
        print(f"[ERROR] Conpot log file not found: {logfile_path}")
    except Exception as e:
        print(f"[ERROR] Tailing conpot.log failed: {e}")
    print("[TAIL] Stopped tailing conpot.log")

# Main
def main():
    stop_event = threading.Event()
    threads = []

    # synthetic generator thread
    if GENERATE_SYNTHETIC:
        t = threading.Thread(target=synthetic_generator, args=(stop_event,), daemon=True)
        threads.append(t)
        t.start()

    # conpot tail thread
    if FORWARD_CONPOT_LOGS:
        t2 = threading.Thread(target=tail_conpot_forwarder, args=(stop_event, LOG_FILE), daemon=True)
        threads.append(t2)
        t2.start()

    # run for configured duration (or forever)
    try:
        if RUN_DURATION_MIN and RUN_DURATION_MIN > 0:
            # convert minutes to seconds
            end_time = time.time() + (RUN_DURATION_MIN * 60)
            while time.time() < end_time and not stop_event.is_set():
                time.sleep(1)
        else:
            # run forever until keyboard interrupt
            while not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user, shutting down...")
    finally:
        stop_event.set()
        # give threads a moment to finish
        time.sleep(1)
        print("[MAIN] Exited.")

if __name__ == "__main__":
    main()