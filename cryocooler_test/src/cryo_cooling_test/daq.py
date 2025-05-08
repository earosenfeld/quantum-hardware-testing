import random
import time

class DAQ:
    def __init__(self, sensors, network_latency=0.0, fail_rate=0.0):
        self.sensors = sensors
        self.network_latency = network_latency
        self._fail_rate = fail_rate
        self.connected = True
        self._last_fail_check = None

    @property
    def fail_rate(self):
        return self._fail_rate
    
    @fail_rate.setter
    def fail_rate(self, value):
        self._fail_rate = value
        self._last_fail_check = None  # Reset failure check on rate change

    def read_all(self):
        """Read data from all sensors."""
        if self.network_latency > 0:
            time.sleep(self.network_latency)
        
        # Only check for failure once per read cycle
        if self._last_fail_check is None:
            self._last_fail_check = random.random()
            
        if not self.connected or self._last_fail_check < self.fail_rate:
            self.connected = False
            raise ConnectionError("DAQ network failure")
            
        return [sensor.read() for sensor in self.sensors]
    
    def reconnect(self):
        """Attempt to reconnect to the DAQ."""
        self.connected = True
        self._last_fail_check = None 