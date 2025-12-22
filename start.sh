#!/usr/bin/env bash
echo "[+] Starting DMZ services"

cd ~/conpot && docker-compose up -d
cd ~/thingsboard && docker-compose up -d

echo "[+] Starting CALDERA"
cd ~/caldera
source caldera-env/bin/activate
python3 server.py