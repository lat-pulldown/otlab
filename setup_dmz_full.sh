#!/usr/bin/env bash
set -e

echo "======================================="
echo "[+] DMZ FULL SETUP: Conpot → ThingsBoard → CALDERA"
echo "======================================="

########################################
# 0. Base system setup
########################################
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  git make wget curl \
  python3-pip python3-venv \
  docker.io docker-compose

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

########################################
# CONPOT
########################################
########################################
# 2. Create working directory
########################################
echo "[+] Creating conpot working directory"
mkdir -p ~/conpot
cd ~/conpot

########################################
# 3. Create Dockerfile (exact behavior)
########################################
echo "[+] Writing Dockerfile"

cat << 'EOF' > Dockerfile
# Stage 1: Build stage
FROM python:3.8 AS conpot-builder

RUN apt-get update && apt-get install -y \
    git build-essential gcc g++ make \
    libffi-dev libssl-dev libxml2 libxslt1.1 libxslt-dev \
    && pip install --upgrade pip setuptools wheel cython \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/conpot

RUN git clone https://github.com/mushorg/conpot.git . && \
    pip install "gevent==22.10.2" && \
    pip install --no-cache-dir .

RUN adduser --disabled-password --gecos "" conpot && \
    mkdir -p /var/log/conpot && chown -R conpot:conpot /var/log/conpot

RUN mkdir -p /usr/local/lib/python3.8/site-packages/conpot/tests/data/data_temp_fs/ftp && \
    mkdir -p /usr/local/lib/python3.8/site-packages/conpot/tests/data/data_temp_fs/tftp && \
    chmod -R 777 /usr/local/lib/python3.8/site-packages/conpot/tests/data

USER conpot
ENV PATH=$PATH:/home/conpot/.local/bin
ENV USER=conpot

EXPOSE 80 502 161/udp
ENTRYPOINT ["conpot"]
CMD ["--template", "default", "--logfile", "/var/log/conpot/conpot.log", "-f", "--temp_dir", "/tmp"]
EOF

########################################
# 4. docker-compose.yml (exact ports)
########################################
echo "[+] Writing docker-compose.yml"

cat << 'EOF' > docker-compose.yml
version: "3.8"

services:
  conpot:
    build: .
    container_name: conpot
    ports:
      - "80:8800"
      - "502:5020"
      - "161:16100/udp"
    volumes:
      - ./logs:/var/log/conpot
    restart: unless-stopped
EOF

########################################
# 5. Makefile (workflow parity)
########################################
echo "[+] Writing Makefile"

cat << 'EOF' > Makefile
.PHONY: docker build-docker run-docker format

build-docker:
	docker build -t conpot:latest .

run-docker:
	docker run --rm -it \
		-p 80:8800 \
		-p 102:10201 \
		-p 502:5020 \
		-p 161:16100/udp \
		-p 47808:47808/udp \
		-p 623:6230/udp \
		conpot:latest

format:
	black .
EOF

########################################
# 5.5. modbus.xml
########################################
echo "[+] Writing new_modbus.xml"

cat > new_modbus.xml <<EOF
<modbus enabled="True" host="0.0.0.0" port="5020">
    <device_info>
        <VendorName>Default</VendorName>
        <ProductCode>Default</ProductCode>
        <MajorMinorRevision>1.0</MajorMinorRevision>
    </device_info>

    <mode>tcp</mode>
    <delay>50</delay>

    <slaves>
        <slave id="1">
            <blocks>
                <block name="temperature_data">
                    <type>HOLDING_REGISTERS</type>
                    <starting_address>0</starting_address>
                    <size>2</size>
                    <content>memoryModbusSlave1BlockA</content>
                </block>
            </blocks>
        </slave>
    </slaves>
</modbus>
EOF

########################################
# 6. Create log directory
########################################
echo "[+] Creating log directory"
mkdir -p logs

########################################
# 7. Build and run Conpot
########################################
echo "[+] Building Conpot Docker image"
docker-compose build

echo "[+] Starting Conpot"
docker-compose up -d

########################################
# 8. Verification
########################################
echo "[+] Verifying container status"
docker ps

echo "[+] Listening ports"
sudo ss -tulnp | grep -E ':(80|502|161)'

echo "[+] Conpot DMZ setup completed"

########################################
# THINGSBOARD
########################################
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

########################################
# CALDERA
########################################
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