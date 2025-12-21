# Research Objective
This research establishes a comprehensive framework for emulating industrial environments to collect realistic OT (Operational Technology) communication logs and evaluate machine learning models for real-time anomaly detection. By integrating automated attack emulation `Caldera` with a virtualized industrial target `Conpot`, the system generates high-fidelity datasets that represent both normal operations and diverse cyber-attack scenarios. The framework serves as a standardized benchmarking platform to assess how different deep learning architectures handle the strict accuracy and low-latency requirements essential for protecting critical infrastructure.   
## The Objective
The primary objective is to bridge the gap between IT and OT security by:
- Creating a Realistic Testbed: Establishing a reproducible environment to collect authentic industrial logs that reflect cyber-physical correlations.  
- Benchmarking Deep Learning Models: Evaluating various architectures - `Isolation Forest`, `1D-CNN`, `DeepLog`, and `CNN-Transformer Hybrid` - to determine which provides the best balance of detection accuracy, latency, and computational intesity required for critical infrastructure.  
- Cyber-Physical Correlation: Provide a unified dashboard `Thingsboard`, to visualize network attack intensity alongside physical data to better understand the potential impact of cyber events on physical assets.

**Visit [here](https://github.com/lat-pulldown/otlab) for the GitHub Repo.**

---

## Research Walkthrough
Watch this screen recording to see the full flow of log generation, alignment, and model prediction.

<video width="100%" controls>
  <source src="videos/setup_recording.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## System Topology
The system spans two environments. Ensure they are on the same network subnet to allow Modbus communication.

* **Local Machine (macOS/Windows):** Runs the Preprocessor, Caldera, and the Deep-Learning Models.  
* **Virtual Machine (Ubuntu/Linux):** Runs the ISC Honeypot `Conpot` and IoT Platform `Thingsboard`.  
* **Physical OT Device:** Optional. The example Temperature data `temp.csv` is from a Thermal Camera ([OMRON K6PM](https://automation.omron.com/en/us/products/family/K6PM/k6pm-thmd-eip)). If connecting in real-time, be sure they are in the same network.  

---

## Environment Setup

### 1. Getting Started
#### 1.1. Prerequisites
- This setup guide is for macOS (Apple Silicon).
- Also works with intel macbooks and Windows PC (Each commands may be different).  
#### 1.2. Install [Python](https://www.python.org/downloads/)
#### 1.3. Clone [Github Repo](https://github.com/lat-pulldown/otlab)
```
git clone https://github.com/lat-pulldown/otlab.git
```

### 2. Virtual Machine Configuration
#### 2.1. Install [Multipass](https://canonical.com/multipass)
#### Import OVA to Multipass (VM name: dmz, OVA name: cloud-config.yaml)
```
multipass launch --name dmz --cloud-init cloud-config.yaml
```
#### 2.2. Check VM IP Address (We will need this later)
```
multipass list
```
and 
```
multipass info dmz
```
#### 2.3. Start VM
```
multipass start dmz
``` 
```
multipass shell dmz
```
*Which one?
#### 2.4. Launch Conpot and Thingsboard
```
sudo ~/start.sh
```
`sudo ~/stop.sh` to stop
#### 2.5. Open Thingsboard WebUI
Visit http://YOUR_VM_IP:8080  
#### 2.6. Open Caldera-OT
```
python server.py
```
Visit http://YOUR_VM_IP:8888/login  

### 3. Local Machine Setup
#### 3.1. Navigate to otlab
```
cd otlab
```
#### 3.2. Python libarary dependencies
```
pip install -r requirements.txt
```
#### 3.3. Mount folder to transfer logs from Conpot to Local  
```
mkdir otlab/logshare
cd logshare
multipass mount ./logshare dmz:/home/ubuntu/shared
```	

---

## Execution Steps

### 1. Log Generation
#### 1.1. Normal Log (For Training)
**In local environment...**  
1.1.1. Run Normal Polling script
```
cd /otlab/script
python3 robust_polling.py
```
**In VM environment...**  
1.1.2. Move log from Conpot to Local
```
sudo mv /home/ubuntu/conpot/logs/conpot.log /home/ubuntu/conpot/logs/normal.log
```
```
cd ~/conpot
docker compose restart conpot
```
```
sudo mv /home/ubuntu/conpot/logs/nomral.log /home/ubuntu/shared
```
#### 1.2. Noise Log 
**In local environemnt...**  
1.2.1. Run Normal Polling script (Don't run this for `pure_noise.log`)
```
cd /otlab/script
python3 robust_polling.py
```
1.2.2. Open Port:502 and open SSH Tunnel (In a new terminal)
```
sudo ssh -i /var/root/Library/Application\ Support/multipassd/ssh-keys/id_rsa -L 0.0.0.0:50502:localhost:502 ubuntu@<YOUR VM IP>
```
**In VM environment...**  
1.2.3. Check if Port 502 is open
```
nc -zv <YOUR VM IP> 502
```
1.2.4. Move log from Conpot to Local
```
sudo mv /home/ubuntu/conpot/logs/conpot.log /home/ubuntu/conpot/logs/noise.log
```
```
cd ~/conpot
docker compose restart conpot
```
```
sudo mv /home/ubuntu/conpot/logs/noise.log /home/ubuntu/shared
```
#### 1.3. Attack Log 
**In local environment...**  
1.3.1. Run Normal Polling script (Don't run this for `pure_attack.log`)
```
cd /otlab/script
python3 robust_polling.py
```	
1.3.2. Start Caldera
```
cd /otlab
python3 server.py
```
1.3.3. Open WebUI at [http://localhost:8080](http://localhost:8080)  
1.3.4. Create Agent  
1.3.5. Create Operations and run it  
*Local or inside VM???
**In VM environment...**  
1.3.5. Move log from Conpot to Local
```
sudo mv /home/ubuntu/conpot/logs/conpot.log /home/ubuntu/conpot/logs/attack.log
```
```
cd ~/conpot
docker compose restart conpot
```
```
sudo mv /home/ubuntu/conpot/logs/attack.log /home/ubuntu/shared
```
#### 1.4. Mix Log
**In local environment...**  
1.4.1. Put `pure_noise.log` and `pure_attack.log` under `/script`  
1.4.2. Run logmixer.py
```
cd /otlab/script
python3 logmixer.py
```
1.4.3. Align the datetime of `pre_mix.log` to output `mix.log`
```
python3 aligner.py pre_mix.log mix.log
```
1.4.4. Move the created logs `normal.log` `noise.log` `attack.log` `mix.log` to `/preprocessor`  

### 2. Thingsboard
#### 2.1. Send temperature data to Thingsboard
**In local environment...**  
Send tempurature via `temp.csv` taken from a thermal camera
```
cd /otlab/templog
python3 camera_replay.py
```
**In VM environment...**  
#### 2.2. Send Conpot logs in real-time
```
cd /home/ubuntu
python pottotb.py
```  

### 3. Preprocessor  
#### 3.1. Generate training dataset `normal.log`
```
cd /otlab/preprocessor
python3 parser.py -mode train -log normal.log
```
#### 3.2. Generate testing dataset - `noise.log` (Change X according to version)
```
python3 parser.py --mode predict --log noise.log --out noiseX.csv --out_tf noiseX
```
#### 3.3. Generate testing dataset - `attack.log` (Change X according to version)
```
python3 parser.py --mode predict --log attack.log --out attackX.csv --out_tf attackX
```
#### 3.4. Generate testing dataset - `mix.log` (Change X according to version)
```
python3 parser.py --mode predict --log mix.log --out mixX.csv --out_tf mixX
```  

### 4: Evaluating models
**In local environment...**  
#### 4.1. Isolation Forest
```
cd /otlab/iforest
```
Train
```
python3 iforest.py -mode train		
```	
Test for `noise_tf01.csv`
```
python3 iforest.py -mode test -data noise_tf01.csv		
```	
Test for `attack_tf01.csv`
```
python3 iforest.py -mode test -data attack_tf01.csv		
```
Test for `mix.csv`
```
python3 iforest.py -mode test -data mix_tf01.csv		
```		
#### 4.2. 1D-CNN
```
cd /otlab/cnn	
```
Train
```
python3 cnn_train.py		
```	
Test for `noise.csv`
```
python3 cnn.py -mode test -data noise.csv		
```	
Test for `attack.csv`
```
python3 cnn.py -mode test -data attack.csv		
```	
Test for `mix.csv`
```
python3 cnn.py -mode test -data mix.csv		
```
#### 4.3. DeepLog ([GitHub](https://github.com/wuyifan18/DeepLog))
```
cd /otlab/deeplog
```	
Train
```
python3 model_train.py	
```	
Test for `noise.csv`
```
python3 model_test.py -mode test -data noise.csv		
```
Test for `attack.csv`
```
python3 model_test.py -mode test -data attack.csv		
```
Test for `mix.csv`
```
python3 model_test.py -mode test -data mix.csv		
```
#### 4.4. Hybrid Variate
```
cd /otlab/hyvar	
```	
##### 4.4.1. 1D-CNN-Transformer
Train
```
python3 hybrid_train.py
```
Test for `noise.csv`
```
python3 hybrid_test.py -mode test -data noise.csv		
```
Test for `attack.csv`
```
python3 hybrid_test.py -mode test -data attack.csv		
```
Test for `mix.csv`
```
python3 hybrid_test.py -mode test -data mix.csv		
```
##### 4.4.2. Temperature-Variate
Train
```
python3 var_train.py		
```
Test for `noise_tf01.csv`
```
python3 var_test.py -mode test -data noise_tf01.csv		
```
Test for `attack_tf01.csv`
```
python3 var_test.py -mode test -data attack_tf01.csv		
```
Test for `mix_tf01.csv`
```
python3 var_test.py -mode test -data mix_tf01.csv		
```
##### 4.4.3. Correlation Test
Test for `noise.csv`, `noise_tf01.csv`
```
python3 fusion_test.py -cyber noise.csv -phys noise_tf01.csv	
```
Test for `attack.csv`, `attack_tf01.csv`
```
python3 fusion_test.py -cyber attack.csv -phys attack_tf01.csv	
```
Test for `mix.csv`, `mix_tf01.csv`
```
python3 fusion_test.py -cyber mix.csv -phys mix_tf01.csv	
```
