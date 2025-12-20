---
layout: default
title: OTLab User Manual
---

# OTLab: Modbus Honeypot & Anomaly Detection
Welcome to the setup guide for the OTLab research project. This system integrates a **Conpot Honeypot**, **Caldera Bot** for attack simulation, and a **DeepLog-based LSTM model** for anomaly detection.

---

## Project Walkthrough
Watch this screen recording to see the full flow of log generation, alignment, and model prediction.

<video width="100%" controls>
  <source src="videos/setup_recording.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## System Topology
The system spans two environments. Ensure they are on the same network subnet to allow Modbus communication.

* **Local Machine (macOS):** Runs the `robust_polling.py` (normal traffic) and `model_test.py` (AI detection).
* **Virtual Machine (Ubuntu/Linux):** Runs the `Conpot` Honeypot and `caldera_bot`.

---

## Environment Setup

### 1. Getting Started
* Using macOS (Apple Silicon)

* Install [Python](https://www.python.org/downloads/)

* **Clone Github Repo**
```
git clone https://github.com/lat-pulldown/otlab.git
```


### 2. Virtual Machine Configuration
* Install [Multipass](https://canonical.com/multipass)

* Import OVA to Multipass (vm name: dmz, ova folder name: cloud-config.yaml)
```
multipass launch --name research-vm --cloud-init cloud-config.yaml
```
    
* **Check Multipass IP Address**
```
multipass list
```
    
* **Start VM**
```
multipass start dmz
``` 
```
multipass shell dmz
```


### 3. Local Machine Setup
* **Navigate to otlab**
```
cd otlab
```
    
* **Python Dependencies**
```
pip install -r requirements.txt
```
    
* Mount folder to transfoer logs from Conpot to Local

** In VM environment, in /home/ubuntu
```
mkdir shared
```
    
** In local environment
```
mkdir otlab/logshare
cd logshare
multipass mount ./logshare dmz:/home/ubuntu/shared
```

---

## Thingsboard

### 1. Send temperature data to Thingsboard
* In local environment (Sends tempurature via temp.csv taken from a thermal camera)
```
cd /otlab/templog
python3 camera_replay.py
```
		
* In vm environment (Sends Conpot logs in realtime)
```
cd /home/ubuntu
python pottotb.py
```
		
---

## Execution Steps

### 1. Log Generation
* Normal Log (For training data)

  ** In local environment
```
cd /otlab/script
python3 robust_polling.py
```
		
  ** In VM environment (To move log from Conpot to Local)
```
sudo mv /home/ubuntu/conpot/logs/conpot.log /home/ubuntu/conpot/logs/normal.log
cd ~/conpot
docker compose restart conpot
sudo mv /home/ubuntu/conpot/logs/nomral.log /home/ubuntu/shared
```

* Noise Log 

** In local environemnt (Don't run this for pure noise log)
```
cd /otlab/script
python3 robust_polling.py
```
		
** Open Port:502 and open SSH Tunnel
```
cd /otlab
sudo ssh -i /var/root/Library/Application\ Support/multipassd/ssh-keys/id_rsa -L 0.0.0.0:50502:localhost:502 ubuntu@<CONPOT VM IP>
```
		
** In VM environment (To move log from Conpot to Local)
```
sudo mv /home/ubuntu/conpot/logs/conpot.log /home/ubuntu/conpot/logs/noise.log
cd ~/conpot
docker compose restart conpot
sudo mv /home/ubuntu/conpot/logs/noise.log /home/ubuntu/shared
```
		
* Attack Log 

** In local environment (Don't run this for pure attack log)
```
cd /otlab/script
python3 robust_polling.py
```
		
** Open Caldera
```
cd /otlab
python3 server.py
```

Open WebUI at [http://localhost:8080](http://localhost:8080)
		
Create Agent
		
Create Operations and run it 
		
** In VM environment (To move log from Conpot to Local)
```
sudo mv /home/ubuntu/conpot/logs/conpot.log /home/ubuntu/conpot/logs/attack.log
cd ~/conpot
docker compose restart conpot
sudo mv /home/ubuntu/conpot/logs/attack.log /home/ubuntu/shared
```

* Mix Log

** In local environment (Put pure_noise.log and pure_attack.log to /script)
```
cd /otlab/script
python3 logmixer.py
```
Align the datetime of `pre_mix.log` to output `mix.log`
```
python3 aligner.py pre_mix.log mix.log
```

* Move the created logs `normal.log` `noise.log` `attack.log` `mix.log` to /preprocessor
	
		
### 2. Preprocessor
* Generate training dataset `normal.log`
```
cd /otlab/preprocessor
python3 parser.py -mode train -log normal.log
```
		
* Generate testing dataset (Change X according to version) 

** For `noise.log`
```
python3 parser.py --mode predict --log noise.log --out noiseX.csv --out_tf noiseX
```
		
** For `attack.log`
```
python3 parser.py --mode predict --log attack.log --out noiseX.csv--out_tf attackX
```
		
** For `mix.log`
```
python3 parser.py --mode predict --log mix.log --out mixX.csv--out_tf mixX
```

### 3: Evaluating models (In local environment)
* Isolation Forest
```
cd /otlab/iforest
```
		
** Train
```
python3 iforest.py -mode train		
```
		
** Test for `noise_tf01.csv`
```
python3 iforest.py -mode test -data noise_tf01.csv		
```
		
** Test for `attack_tf01.csv`
```
python3 iforest.py -mode test -data attack_tf01.csv		
```
		
** Test for `mix.csv`
```
python3 iforest.py -mode test -data mix_tf01.csv		
```
		
* 1D-CNN
```
cd /otlab/cnn	
```
		
** Train
```
python3 cnn_train.py		
```
		
** Test for `noise.csv`
```
python3 cnn.py -mode test -data noise.csv		
```
		
** Test for `attack.csv`
```
python3 cnn.py -mode test -data attack.csv		
```
		
** Test for `mix.csv`
```
python3 cnn.py -mode test -data mix.csv		
```
		
* DeepLog
```
cd /otlab/deeplog
```
		
** Train
```
python3 model_train.py	
```
		
** Test for `noise.csv`
```
python3 model_test.py -mode test -data noise.csv		
```
		
** Test for `attack.csv`
```
python3 model_test.py -mode test -data attack.csv		
```
		
** Test for `mix.csv`
```
python3 model_test.py -mode test -data mix.csv		
```
		
* Hybrid Variate
```
cd /otlab/hyvar	
```
		
** 1D-CNN-Transformer

** Train
```
python3 hybrid_train.py
```
		
** Test for `noise.csv`
```
python3 hybrid_test.py -mode test -data noise.csv		
```
		
** Test for `attack.csv`
```
python3 hybrid_test.py -mode test -data attack.csv		
```
		
** Test for `mix.csv`
```
python3 hybrid_test.py -mode test -data mix.csv		
```
		
** Temperature-Variate

** Train
```
python3 var_train.py		
```
		
** Test for `noise.csv`
```
python3 var_test.py -mode test -data noise_tf01.csv		
```
		
** Test for `attack.csv`
```
python3 var_test.py -mode test -data attack_tf01.csv		
```
		
** Test for `mix.csv`
```
python3 var_test.py -mode test -data mix_tf01.csv		
```
		
** Correlation Test

** Test for `noise.csv`
```
python3 fusion_test.py -cyber noise.csv -phys noise_tf01.csv	
```
		
** Test for `attack.csv`
```
python3 fusion_test.py -cyber attack.csv -phys attack_tf01.csv	
```
		
** Test for `mix.csv`
```
python3 fusion_test.py -cyber mix.csv -phys mix_tf01.csv	
```
