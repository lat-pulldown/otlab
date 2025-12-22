#!/usr/bin/env bash
set -e

echo "[+] Starting CALDERA DMZ setup (faithful reconstruction)"

########################################
# 0. System dependencies
########################################
echo "[+] Installing system dependencies"
sudo apt update
sudo apt install -y git python3-pip python3-venv wget curl

########################################
# 1. Clone CALDERA (recursive)
########################################
echo "[+] Cloning CALDERA repository"
git clone https://github.com/mitre/caldera.git --recursive

cd caldera

########################################
# 2. Python virtual environment
########################################
echo "[+] Creating Python virtual environment"
python3 -m venv caldera-env

source caldera-env/bin/activate

########################################
# 3. Install Python requirements (inside venv)
########################################
echo "[+] Installing Python requirements"
pip3 install -r requirements.txt

########################################
# 4. Install Modbus plugin
########################################
echo "[+] Installing Modbus plugin"
cd plugins
git clone https://github.com/mitre/modbus.git
cd ..

########################################
# 5. Manual config step (cannot be automated faithfully)
########################################
echo "======================================================"
echo "[!] MANUAL STEP REQUIRED"
echo "    Edit conf/local.yml"
echo "    Under 'plugins:' add:"
echo "        - modbus"
echo ""
echo "    Press ENTER when finished."
echo "======================================================"
read

########################################
# 6. Node.js 20 installation
########################################
echo "[+] Installing Node.js 20.x"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

echo "[+] Node.js version:"
node -v

########################################
# 7. Go removal (if exists)
########################################
echo "[+] Removing existing Go installation (if any)"
sudo apt-get remove -y golang-go || true
sudo rm -rf /usr/local/go || true

########################################
# 8. Install Go 1.21
########################################
echo "[+] Installing Go 1.21"
wget https://go.dev/dl/go1.21.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.linux-amd64.tar.gz

export PATH=$PATH:/usr/local/go/bin

echo "[+] Go version:"
go version

########################################
# 9. Build CALDERA (first run only)
########################################
echo "[+] Building CALDERA (first run)"
python3 server.py --build