---
layout: default
title: OTLab User Manual
---

# 🛡️ OTLab: Modbus Honeypot & Anomaly Detection
Welcome to the setup guide for the OTLab research project. This system integrates a **Conpot Honeypot**, **Caldera Bot** for attack simulation, and a **DeepLog-based LSTM model** for anomaly detection.

---

## 📺 Project Walkthrough
Watch this screen recording to see the full flow of log generation, alignment, and model prediction.

<video width="100%" controls>
  <source src="videos/setup_recording.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 🌐 System Topology
The system spans two environments. Ensure they are on the same network subnet to allow Modbus communication.

* **Local Machine (macOS):** Runs the `robust_polling.py` (normal traffic) and `model_test.py` (AI detection).
* **Virtual Machine (Ubuntu/Linux):** Runs the `Conpot` Honeypot and `caldera_bot`.



---

## 🛠️ Environment Setup

### 1. Virtual Machine Configuration
* **Network:** Set to `Bridged Adapter` or `Host-Only` in your VM settings.
* **Conpot:** Ensure Docker is running.
    ```bash
    docker run -it -p 5020:5020 conpot/conpot
    ```

### 2. Local Machine Setup
* **Python Dependencies:**
    ```bash
    pip install torch pandas psutil numpy pymodbus
    ```

---

## 🚀 Execution Steps

### Phase 1: Data Generation
To generate the `attack.log`, run these simultaneously:
1.  **Start Polling:** `python robust_polling.py`
2.  **Start Attack:** Trigger `caldera_bot` from the VM.

### Phase 2: Log Alignment
Before feeding data to the model, use the aligner to fix timestamp gaps:
```bash
python aligner.py attack.log cleaned_attack.log