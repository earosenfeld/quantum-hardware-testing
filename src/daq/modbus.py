"""Simulated instrument-link fault injectors.

These are NOT protocol implementations — no Modbus framing or VISA
session management lives here. They model the *failure behavior* of an
instrument link (probabilistic drops, latency, reconnect cycles) so the
DAQ layer's error handling and recovery paths can be exercised in tests
without hardware.
"""

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
