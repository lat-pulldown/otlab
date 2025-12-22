#!/usr/bin/env bash
set -e

echo "[+] Starting Conpot DMZ setup"

########################################
# 0. System update
########################################
echo "[+] Updating system"
sudo apt update && sudo apt upgrade -y

########################################
# 1. Install Docker & docker-compose
########################################
echo "[+] Installing Docker"
sudo apt install -y docker.io docker-compose make git

sudo systemctl enable docker
sudo systemctl start docker

sudo usermod -aG docker $USER

echo "[!] Docker group updated. Re-login may be required."
sleep 2

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