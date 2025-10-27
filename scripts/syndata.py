# syndata.py
import random
import datetime
import json

def generate_log_entry():
    """1件の疑似ログを生成"""
    timestamp = datetime.datetime.now().isoformat()
    log_level = random.choice(["INFO", "WARNING", "ERROR", "DEBUG"])
    component = random.choice(["sensor", "actuator", "network", "modbus", "conveyor"])
    message = random.choice([
        "Operation successful",
        "Value out of range",
        "Connection lost",
        "Retrying command",
        "Temperature threshold exceeded",
        "Pressure within limit",
    ])
    
    return {
        "timestamp": timestamp,
        "level": log_level,
        "component": component,
        "message": message
    }

def generate_log_file(file_path="synthetic_logs.json", n=1000):
    """複数の疑似ログを生成してファイルに保存"""
    logs = [generate_log_entry() for _ in range(n)]
    with open(file_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"{n}件の疑似ログを {file_path} に生成しました。")

if __name__ == "__main__":
    generate_log_file()