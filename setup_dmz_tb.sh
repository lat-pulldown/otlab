#!/usr/bin/env bash
set -e

echo "[+] Starting ThingsBoard CE DMZ setup"

########################################
# 0. System update
########################################
sudo apt update && sudo apt upgrade -y

########################################
# 1. Install Docker & docker-compose
########################################
sudo apt install -y docker.io docker-compose

sudo systemctl enable docker
sudo systemctl start docker

sudo usermod -aG docker $USER

echo "[!] Docker group updated. Re-login may be required."
sleep 2

########################################
# 2. Create ThingsBoard directory
########################################
mkdir -p ~/thingsboard
cd ~/thingsboard

########################################
# 3. Write docker-compose.yml (exact)
########################################
echo "[+] Writing docker-compose.yml"

cat << 'EOF' > docker-compose.yml
version: "3.8"

services:
  postgres:
    restart: always
    image: "postgres:16"
    ports:
      - "5432"
    environment:
      POSTGRES_DB: thingsboard
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres-data:/var/lib/postgresql/data

  thingsboard-ce:
    restart: always
    image: "thingsboard/tb-node:4.2.1"
    ports:
      - "8080:8080"
      - "7070:7070"
      - "1883:1883"
      - "8883:8883"
      - "5683-5688:5683-5688/udp"
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
    environment:
      TB_SERVICE_ID: tb-ce-node
      SPRING_DATASOURCE_URL: jdbc:postgresql://postgres:5432/thingsboard
    depends_on:
      - postgres

volumes:
  postgres-data:
    name: tb-postgres-data
    driver: local
EOF

########################################
# 4. Start ThingsBoard
########################################
echo "[+] Starting ThingsBoard stack"
docker-compose up -d

########################################
# 5. Verification
########################################
docker ps

echo "[+] Waiting for ThingsBoard to initialize..."
echo "    (first startup may take 2–3 minutes)"

sleep 10

echo "[+] Open browser at:"
echo "    http://<DMZ_VM_IP>:8080"

echo "[+] Default credentials:"
echo "    sysadmin@thingsboard.org / sysadmin"