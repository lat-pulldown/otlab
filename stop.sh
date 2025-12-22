#!/usr/bin/env bash
echo "[+] Stopping DMZ services"

cd ~/conpot && docker-compose down
cd ~/thingsboard && docker-compose down

echo "[+] CALDERA must be stopped manually (Ctrl+C)"