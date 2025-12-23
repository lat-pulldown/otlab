**Links to Repository: [otlab](https://github.com/lat-pulldown/otlab) (local environment setup), [vm-dmz](https://github.com/lat-pulldown/vm-dmz) (vm environment setup)**

## Research Overview
This research aims to establish a framework for emulating industrial environments to collect realistic OT (Operational Technology) communication logs and evaluate machine learning models for real-time anomaly detection. By integrating automated attack emulation `Caldera` with a virtualized industrial target `Conpot`, the system generates high-fidelity datasets that represent both normal operations and diverse cyber-attack scenarios. The framework serves as a standardized benchmarking platform to assess how different deep learning architectures handle the strict accuracy and low-latency requirements essential for protecting critical infrastructure.   


## Objectives
- **Creating a Realistic Testbed:** To establish a reproducible environment to collect authentic industrial logs.  
- **Benchmarking Deep Learning Models:** To evaluat various architectures - `Isolation Forest`, `1D-CNN`, `DeepLog`, and `CNN-Transformer Hybrid` - to determine which provides the best balance of detection accuracy, latency, and computational load for critical infrastructure.  
- **Cyber-Physical Visualization and Modeling:** To provide a dashboard `Thingsboard`, to visualize network attack intensity alongside physical data to better understand the potential impact of cyber events on physical assets. To implement a `CNN-Transformer Hybrid` model that performs feature fusion across multi-domain datasets to improve detection accuracy and interpret cyber-physical correlation.

---

## Walkthrough
Watch this screen recording to see the full flow of log generation, alignment, and model prediction.

<video width="100%" controls>
  <source src="videos/setup_recording.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## System Topology
The system spans two environments. Ensure they are on the same network subnet to allow Modbus communication.

* **Local Machine (macOS/Windows):** Runs the Preprocessor, scripts, datasets, and the Deep-Learning Models.  
* **Virtual Machine (Ubuntu/Linux):** Runs the ICS honeypot `Conpot`, IoT platform `Thingsboard`, and attack emulation tool `Caldera`.  
* **Physical OT Device:** Optional. The example Temperature data `temp.csv` is from a Thermal Camera ([OMRON K6PM](https://automation.omron.com/en/us/products/family/K6PM/k6pm-thmd-eip)). If connecting in real-time, be sure they are in the same network subnet.  

---

## Environment Setup

### 1. Getting Started
#### 1.1. Prerequisites
- This setup guide is for macOS (Apple Silicon).
- Also works with Intel Macs and Windows PC (Each commands may be different).  
  
#### 1.2. Install [Python](https://www.python.org/downloads/)

### 2. Virtual Machine Configuration
#### 2.1. Install [Multipass](https://canonical.com/multipass)
From Homebrew Terminal
```
brew install --cask multipass
```
Verify with `multipass version` and `multipass list`
#### 2.2. Create a VM (We will name it dmz)
```
multipass launch 22.04 \
  --name dmz \
  --cpus 4 \
  --memory 8G \
  --disk 40G
```
#### 2.4. Enter the shell
```
multipass shell dmz
```
`multipass stop dmz` to stop, `multipass start dmz` to start again.
#### 2.5. (Once inside the shell...) Clone Github [vm-dmz](https://github.com/lat-pulldown/vm-dmz)
```
git clone https://github.com/lat-pulldown/vm-dmz.git
```
#### 2.6. Build Conpot, Thingsboard, and Caldera (all at once)
```
chmod +x setup_dmz_full.sh
./setup_dmz_full.sh
```
For individual setup use `setup_dmz_conpot`, `setup_dmz_tb`, or `setup_dmz_caldera`. Make sure to use `chmod +x setup_dmz_xxxx.sh`.
#### 2.7. Open Thingsboard WebUI
1. Visit http://VM_IP:8080 in your local environment 
2. Log in as usr:`tenant@thingsboard.org` pass: `tenant`
3. Create Device
4. Copy Access Token  

#### 2.8. Change Conpot
1. Edit ~conpot/conpot/testing.cfg 
Create `http_json` and `modbus`
```
[http_json]
enabled = True
host = <VM_IP>
port = 8080
url = /api/v1/<DEVICE_ACCESS_TOKEN>/telemetry
method = POST
interval = 5
```
```
[modbus]
enabled = True
```  
2. Copy the new xml file  
```
cd ~/conpot
docker cp new_modbus.xml conpot:/usr/local/lib/python3.8/site-packages/conpot/templates/default/modbus/modbus.xml
```
#### 2.9. Open Caldera WebUI
Visit http://VM_IP:8888 in your local environment
#### 2.10. To launch Conpot, Thingsboard, and Caldera
```
make start
```
`make stop` to stop  
#### 2.11. Make a folder for sharing logs with local enviornment
```
mkdir ~/shared
```

### 3. Local Machine Setup
#### 3.1. Clone [Github Repo](https://github.com/lat-pulldown/otlab)
```
git clone https://github.com/lat-pulldown/otlab.git && cd otlab
```
#### 3.2. Python libarary dependencies
```
pip3 install -r requirements.txt
```
#### 3.3. Mount folder to transfer logs from Conpot to Local  
```
mkdir otlab/logshare && cd logshare
multipass mount ./logshare dmz:/home/ubuntu/shared
```	

---

## Execution Steps
**In VM environment...** 
### 0. Start all VM services (Starts Conpot, Thingsboard, Caldera, and pottotb.py)
```
sudo ~/start.sh
```
`sudo ~/stop.sh` to stop all.
### 1. Log Generation
#### 1.1. Normal Log for Training (Insert VM_IP to `robust_polling.py` @line 6)
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
sudo ssh -i /var/root/Library/Application\ Support/multipassd/ssh-keys/id_rsa -L 0.0.0.0:50502:localhost:502 ubuntu@<VM_IP>
```
**In VM environment...**  
1.2.3. Check if Port 502 is open
```
nc -zv <VM_IP> 502
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
**In local environment...** 
#### 1.3. Attack Log  
1.3.1. Run Normal Polling script (Don't run this for `pure_attack.log`)
```
cd /otlab/script
python3 robust_polling.py
```	
**In VM environment...**  
1.3.2. Start Caldera (Only if you haven't started it with `sudo ~/start.sh`)
```
cd /home/ubuntu/caldera
python3 server.py
```
1.3.3. Open WebUI at http://<VM_IP>:8888
1.3.4. Create an Agent  
1.3.5. Create Operations and run it  
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
**In local environment...**  
#### 1.4. Mix Log
1.4.1. Put `normal.log`, `pure_noise.log`, and `pure_attack.log` under `/script`  
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
**In local environment...**  
#### 2.1. Send temperature data to Thingsboard
Send tempurature via `temp.csv` taken from a thermal camera (Copy your ACCESS_TOKEN from Thingsboard to `camera_replay.py` @line 13)
```
cd /otlab/templog
python3 camera_replay.py
```
**In VM environment...**  
#### 2.2. Send Conpot logs in real-time (Copy VM_IP and DEVICE_ACCESS_TOKEN from Thingsboard to `pottotb.py` @line 10, 11)
```
cd /home/ubuntu && python3 pottotb.py
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
python3 iforest.py -mode test -data ../data/noise_tf.csv		
```	
Test for `attack_tf01.csv`
```
python3 iforest.py -mode test -data ../data/attack_tf.csv		
```
Test for `mix.csv`
```
python3 iforest.py -mode test -data ../data/mix_tf.csv		
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
python3 cnn.py -mode test -data ../data/noise.csv		
```	
Test for `attack.csv`
```
python3 cnn.py -mode test -data ../data/attack.csv		
```	
Test for `mix.csv`
```
python3 cnn.py -mode test -data ../data/mix.csv		
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
python3 model_test.py -mode test -data ../data/noise.csv		
```
Test for `attack.csv`
```
python3 model_test.py -mode test -data ../data/attack.csv		
```
Test for `mix.csv`
```
python3 model_test.py -mode test -data ../data/mix.csv		
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
python3 hybrid_test.py -mode test -data ../data/noise.csv		
```
Test for `attack.csv`
```
python3 hybrid_test.py -mode test -data ../data/attack.csv		
```
Test for `mix.csv`
```
python3 hybrid_test.py -mode test -data ../data/mix.csv		
```
##### 4.4.2. Temperature-Variate
Train
```
python3 var_train.py		
```
Test for `noise_tf01.csv`
```
python3 var_test.py -mode test -data ../data/noise_tf.csv		
```
Test for `attack_tf01.csv`
```
python3 var_test.py -mode test -data ../data/attack_tf.csv		
```
Test for `mix_tf01.csv`
```
python3 var_test.py -mode test -data ../data/mix_tf.csv		
```
##### 4.4.3. Correlation Test
Test for `noise.csv`, `noise_tf01.csv`
```
python3 fusion_test.py -cyber ../data/noise.csv -phys ../data/noise_tf.csv	
```
Test for `attack.csv`, `attack_tf01.csv`
```
python3 fusion_test.py -cyber ../data/attack.csv -phys ../data/attack_tf.csv	
```
Test for `mix.csv`, `mix_tf01.csv`
```
python3 fusion_test.py -cyber ../data/mix.csv -phys ../data/mix_tf.csv	
```
