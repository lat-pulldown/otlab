.PHONY: all conpot thingsboard caldera deeplog

all: conpot thingsboard caldera deeplog

conpot:
	cd /Users/renshiro/research/data/conpotog && docker-compose up -d

thingsboard:
	cd /Users/renshiro/research/data && docker compose up -d && docker compose logs -f thingsboard-ce

caldera:
	cd /Users/renshiro/research/data/calderaog && python3 server.py

deeplog:
	cd /Users/renshiro/research/ml/deeplogog
