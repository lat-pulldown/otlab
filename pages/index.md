## Research Overview
This research aims to establish a framework for emulating industrial environments to collect realistic OT (Operational Technology) communication logs and evaluate machine learning models for real-time anomaly detection. By integrating automated attack emulation `Caldera` with a virtualized industrial target `Conpot`, the system generates high-fidelity datasets that represent both normal operations and diverse cyber-attack scenarios. The framework serves as a standardized benchmarking platform to assess how different deep learning architectures handle the strict accuracy and low-latency requirements essential for protecting critical infrastructure.   


## Objectives
- **Creating a Realistic Testbed:** To establish a reproducible environment to collect authentic industrial logs.  
- **Benchmarking Deep Learning Models:** To evaluate various architectures - `Isolation Forest`, `1D-CNN`, `DeepLog`, and `CNN-Transformer Hybrid` - to determine which provides the best balance of detection accuracy, latency, and computational load for critical infrastructure.  
- **Cyber-Physical Visualization and Modeling:** To provide a dashboard `ThingsBoard`, to visualize network attack intensity alongside physical data to better understand the potential impact of cyber events on physical assets. To implement a `CNN-Transformer Hybrid` model that performs feature fusion across multi-domain datasets to improve detection accuracy and interpret cyber-physical correlation.

---

## Walkthrough
Watch this screen recording to see the flow of log generation, and model prediction.
<iframe width="560" height="315" src="https://www.youtube.com/embed/4OtfZw2w0F0?si=G2Uu-qFIiDVDH9s7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## System Structure
The system spans two environments. Ensure they are on the same network subnet to allow Modbus communication.

* **Local Machine (macOS/Windows):** Runs the Preprocessor, scripts, datasets, and the Deep-Learning Models.  
* **Virtual Machine (Ubuntu/Linux):** Runs the ICS honeypot `Conpot`, IoT platform `ThingsBoard`, and attack emulation tool `Caldera`.  
* **Physical OT Device:** Optional. The example Temperature data `temp.csv` is from a Thermal Camera ([OMRON K6PM](https://automation.omron.com/en/us/products/family/K6PM/k6pm-thmd-eip)). If connecting in real-time, make sure they are in the same network subnet.  


**Visit [otlab](https://github.com/lat-pulldown/otlab) for local environment setup, and [vm-dmz](https://github.com/lat-pulldown/vm-dmz) for vm environment setup.**

---

## Environment Setup

### 1. Getting Started
#### 1.1. Prerequisites
- This setup guide is for Apple Silicon macOS.
- Commands may be different for Intel Macs and Windows PC.  
  
#### 1.2. Install [Python](https://www.python.org/downloads/)

### 2. Virtual Machine Configuration
#### 2.1. Install [Multipass](https://canonical.com/multipass)
From Homebrew Terminal
```
brew install --cask multipass
```
Verify with `multipass version`
#### 2.2. Create a VM (named `dmz`)
```
multipass launch lts \
  --name dmz \
  --cpus 4 \
  --memory 8G \
  --disk 40G
```
Remember your IP address of VM with `multipass list` (We will call it VM_IP)
#### 2.3. Enter the shell (`exit` to exit shell)
```
multipass shell dmz
```
`multipass stop dmz` to stop, `multipass start dmz` to start it again.
#### 2.4. (Once inside the shell...) Clone Github [vm-dmz](https://github.com/lat-pulldown/vm-dmz)
```
git clone https://github.com/lat-pulldown/vm-dmz.git
```
#### 2.5. Build Conpot, ThingsBoard, and Caldera (Use different terminal for each)
**Note:** This shell script is optimized for Apple Silicon (ARM-based macOS). If you are using an Intel-based Mac, Windows (AMD/Intel), or Linux machine, some commands—especially Docker image tags or architecture-specific settings—may need to be adjusted accordingly.  
For Conpot
```
chmod +x setup_conpot.sh
./setup_conpot.sh
```
For ThingsBoard
```
chmod +x setup_tb.sh
./setup_tb.sh
```
For Caldera
```
chmod +x setup_caldera.sh
./setup_caldera.sh
```
#### 2.6. Open ThingsBoard WebUI
1. Visit http://VM_IP:8080 in a browser   
2. Log in as **username:`tenant@thingsboard.org` password: `tenant`** (You can change it later)
3. Create Device for Conpot and pysical device (If you have one)
4. Copy Device `access token` (We will call it XXX_ACCESS_TOKEN)

#### 2.7. Change Conpot
1. Edit ~/conpot/conpot/testing.cfg 
Create `http_json` and `modbus`
```
[http_json]
enabled = True
host = <VM_IP>
port = 8080
url = /api/v1/<CONPOT_ACCESS_TOKEN>/telemetry
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

#### 2.8. Open Caldera WebUI
Visit http://VM_IP:8888 in your local environment 
#### 2.9. Make a folder for sharing logs with local enviornment
```
mkdir ~/shared
```
#### 2.9. Start all services at once (From VMs root)
```
./start.sh
```
`./stop.sh` to stop all services  
### 3. Local Machine Setup
#### 3.1. Clone [Github Repo](https://github.com/lat-pulldown/otlab)
```
git clone https://github.com/lat-pulldown/otlab.git && cd otlab
```
#### 3.2. Install Python library dependencies
```
pip3 install -r requirements.txt
```
#### 3.3. Mount folder for transfering logs from Conpot to local  
```
mkdir otlab/logshare && cd logshare
multipass mount ./logshare dmz:/home/ubuntu/shared
```	

---

## Execution Steps
**In VM environment...** 
### 1. Start VM services individually (To use different terminals for each)  
**Note:** You can start and stop all services at once using `./start.sh` and `./stop.sh`.  
To start Conpot
```
cd ~/conpot && docker-compose up -d
```
To stop Conpot
```
docker-compose down
```
To start Caldera (`Ctrl+C` to stop)
```
cd ~/caldera
source caldera-env/bin/activate && python3 server.py
```
To start ThingsBoard
```
cd ~/thingsboard && docker-compose up -d
```
To stop ThingsBoard
```
docker-compose down
```
To start pottotb.py (`Ctrl+C` to stop)
```
cd ~ && python3 pottotb.py
```
### 2. Log Generation
#### 2.1. Normal Log for Training (Insert VM_IP to `robust_polling.py` @line 6)
**In local environment...**  
2.1.1. Run Normal Polling script
```
cd /otlab/script
python3 robust_polling.py
```
**In VM environment...**  
2.1.2. Move log from Conpot to Local
```
sudo mv ~/conpot/logs/conpot.log ~/conpot/logs/normal.log
```
```
cd ~/conpot
docker compose restart conpot
```
```
sudo mv ~/conpot/logs/nomral.log ~/shared
```
#### 2.2. Noise Log 
**In local environemnt...**  
2.2.1. Run Normal Polling script (Don't run this for `pure_noise.log`)
```
cd /otlab/script
python3 robust_polling.py
```
2.2.2. Open Port:502 and open SSH Tunnel (In a new terminal)
```
sudo ssh -i /var/root/Library/Application\ Support/multipassd/ssh-keys/id_rsa -L 0.0.0.0:50502:localhost:502 ubuntu@<VM_IP>
```
**In VM environment...**  
2.2.3. Check if Port 502 is open
```
nc -zv <VM_IP> 502
```
2.2.4. Move log from Conpot to Local
```
sudo mv ~/conpot/logs/conpot.log ~/conpot/logs/noise.log
```
```
cd ~/conpot
docker compose restart conpot
```
```
sudo mv ~/conpot/logs/noise.log ~/shared
```
**In local environment...** 
#### 2.3. Attack Log  
2.3.1. Run Normal Polling script (Don't run this for `pure_attack.log`)
```
cd /otlab/script
python3 robust_polling.py
```	
**In VM environment...**  
2.3.2. Start Caldera (If you haven't started it already)
```
cd ~/caldera
python3 server.py
```
2.3.3. Open WebUI at http://VM_IP:8888
2.3.4. Create an Agent  
2.3.5. Create Operations and run it  
2.3.5. Move log from Conpot to Local
```
sudo mv ~/conpot/logs/conpot.log ~/conpot/logs/attack.log
```
```
cd ~/conpot
docker compose restart conpot
```
```
sudo mv ~/conpot/logs/attack.log ~/shared
```
**In local environment...**  
#### 2.4. Mix Log
2.4.1. Put `normal.log`, `pure_noise.log`, and `pure_attack.log` under `/script`  
2.4.2. Run logmixer.py
```
cd /otlab/script
python3 logmixer.py
```
2.4.3. Align the datetime of `pre_mix.log` to output `mix.log`
```
python3 aligner.py pre_mix.log mix.log
```
2.4.4. Move the created logs `normal.log` `noise.log` `attack.log` `mix.log` to `/preprocessor`  

### 3. ThingsBoard
**In local environment...**  
#### 3.1. Send temperature data to ThingsBoard
Send tempurature via `temp.csv` taken from a thermal camera (Copy your PHYSICAL_ACCESS_TOKEN from ThingsBoard to `camera_replay.py` @line 13)
```
cd /otlab/templog
python3 camera_replay.py
```
**In VM environment...**  
#### 3.2. Send Conpot logs in real-time (Copy VM_IP and CONPOT_ACCESS_TOKEN from ThingsBoard to `pottotb.py` @line 10, 11)
```
cd ~ && python3 pottotb.py
```  

### 4. Preprocessor  
#### 4.1. Generate training dataset `normal.log`
```
cd /otlab/preprocessor
python3 parser.py -mode train -log normal.log
```
#### 4.2. Generate testing dataset - `noise.log` (Change X according to version)
```
python3 parser.py --mode predict --log noise.log --out noiseX.csv --out_tf noiseX
```
#### 4.3. Generate testing dataset - `attack.log` (Change X according to version)
```
python3 parser.py --mode predict --log attack.log --out attackX.csv --out_tf attackX
```
#### 4.4. Generate testing dataset - `mix.log` (Change X according to version)
```
python3 parser.py --mode predict --log mix.log --out mixX.csv --out_tf mixX
```  

### 5: Evaluating models
**In local environment...**  
#### 5.1. Isolation Forest
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
#### 5.2. 1D-CNN
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
#### 5.3. DeepLog ([GitHub](https://github.com/wuyifan18/DeepLog))
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
#### 5.4. Hybrid Variate
```
cd /otlab/hyvar	
```	
##### 5.4.1. 1D-CNN-Transformer
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
##### 5.4.2. Temperature-Variate
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
##### 5.4.3. Correlation Test
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

