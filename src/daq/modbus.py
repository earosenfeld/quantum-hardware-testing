import time
import random

class ModbusTCP:
    def __init__(self, fail_rate=0.0, latency=0.0):
        self.fail_rate = fail_rate
        self.latency = latency
        self.connected = True

    def send(self, data):
        if not self.connected or random.random() < self.fail_rate:
            self.connected = False
            print("[ModbusTCP] Communication failure!")
            raise ConnectionError("ModbusTCP communication error")
        time.sleep(self.latency)
        print(f"[ModbusTCP] Sent: {data}")
        return True

    def reconnect(self):
        print("[ModbusTCP] Reconnecting...")
        time.sleep(0.5)
        self.connected = True

class VISA:
    def __init__(self, fail_rate=0.0, latency=0.0):
        self.fail_rate = fail_rate
        self.latency = latency
        self.connected = True

    def send(self, data):
        if not self.connected or random.random() < self.fail_rate:
            self.connected = False
            print("[VISA] Communication failure!")
            raise ConnectionError("VISA communication error")
        time.sleep(self.latency)
        print(f"[VISA] Sent: {data}")
        return True

    def reconnect(self):
        print("[VISA] Reconnecting...")
        time.sleep(0.5)
        self.connected = True
