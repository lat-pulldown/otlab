import time
from pymodbus.client import ModbusTcpClient 
from pymodbus.exceptions import ModbusException

# --- CONFIGURATION ---
TARGET_IP = 'YOUR VM IP'
TARGET_PORT = 502           # Keeping 502 assuming your Docker map is -p 502:5020

# MATCHING YOUR NEW XML:
TARGET_UNIT = 1             # Matches <slave id="1">
TEMP_ADDRESS = 0            # Matches <starting_address>0</starting_address>

# PREVENTS TIMEOUTS:
POLL_INTERVAL = 2.0         

def run_polling():
    print("--- STARTING ROBUST POLLING ---")
    print(f"Target: {TARGET_IP}:{TARGET_PORT}")
    print(f"Unit ID: {TARGET_UNIT}")
    print(f"Register: {TEMP_ADDRESS}")
    print(f"Interval: {POLL_INTERVAL}s")
    print("Press Ctrl+C to stop.")

    client = ModbusTcpClient(TARGET_IP, port=TARGET_PORT)
    client.unit_id = TARGET_UNIT 

    if client.connect():
        print(f"[+] Initial connection successful")
    else:
        print(f"[-] Initial connection failed")

    try:
        while True:
            # AUTO-RECONNECT LOGIC
            if not client.is_socket_open():
                print(f"[!] Connection lost. Reconnecting...")
                if client.connect():
                    print(f"[+] Reconnected!")
                else:
                    time.sleep(POLL_INTERVAL)
                    continue

            try:
                # READ REGISTER 0
                rr = client.read_holding_registers(
                    address=TEMP_ADDRESS, 
                    count=1
                )

                if rr.isError():
                    print(f"[{time.strftime('%H:%M:%S')}] Modbus Error: {rr}")
                else:
                    value = rr.registers[0]
                    print(f"[{time.strftime('%H:%M:%S')}] Success: Register {TEMP_ADDRESS} = {value}")

            except ModbusException as e:
                print(f"[{time.strftime('%H:%M:%S')}] Protocol Exception: {e}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Socket Error: {e}")
                client.close() # Force reconnect next loop
            
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping polling...")
    finally:
        client.close()

if __name__ == "__main__":
    run_polling()