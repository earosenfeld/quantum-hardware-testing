import random

class Sensor:
    def __init__(self, id, fail_rate=0.0):
        self.id = id
        self.fail_rate = fail_rate
        self.failed = False
        self._last_fail_check = None

    def read(self):
        # Only check for failure once per read cycle
        check_value = random.random()
        self._last_fail_check = check_value
        
        if check_value < self.fail_rate and not self.failed:
            self.failed = True
            raise RuntimeError(f"Sensor {self.id} failed")
            
        if self.failed:
            raise RuntimeError(f"Sensor {self.id} is in failed state")
            
        return random.uniform(3.8, 4.2)  # Simulated temperature reading

    def reset(self):
        """Reset the sensor from a failed state."""
        self.failed = False
        self._last_fail_check = None 