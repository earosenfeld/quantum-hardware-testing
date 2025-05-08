import random
import time
import numpy as np
from datetime import datetime

class Sensor:
    """Simulated temperature sensor with realistic behavior and error handling."""
    
    def __init__(self, id, base_temp=4.0, noise_level=0.01, drift_rate=0.0002, 
                 response_time=0.1, fail_rate=0.001):
        """
        Initialize a simulated temperature sensor.
        
        Args:
            id (int): Unique identifier for the sensor
            base_temp (float): Base temperature in Kelvin
            noise_level (float): Standard deviation of temperature noise
            drift_rate (float): Rate of temperature drift per hour
            response_time (float): Sensor response time in seconds
            fail_rate (float): Probability of sensor failure per reading
        """
        self.id = id
        self.base_temp = base_temp
        self.noise_level = noise_level
        self.drift_rate = drift_rate
        self.response_time = response_time
        self.fail_rate = fail_rate
        
        self.failed = False
        self.last_reading = base_temp + random.uniform(-0.05, 0.05)  # Smaller initial offset
        self.last_read_time = 0.0
        self.start_time = time.time()  # Store the start time
        
    def read(self, current_time=None):
        """
        Read the current temperature with realistic noise and drift.
        
        Args:
            current_time (float): Current simulation time in seconds
            
        Returns:
            float: Temperature reading in Kelvin
            
        Raises:
            RuntimeError: If sensor has failed or experiences a random failure
        """
        if current_time is None:
            current_time = time.time() - self.start_time  # Use relative time
        
        # Check for random failure
        if random.random() < self.fail_rate and not self.failed:
            self.failed = True
            raise RuntimeError(f"Sensor {self.id} has failed")
        
        if self.failed:
            raise RuntimeError(f"Sensor {self.id} is in failed state")
        
        # Calculate time-based drift (using simulation time)
        hours_elapsed = current_time / 3600
        drift = self.drift_rate * hours_elapsed
        
        # Add random noise with some periodic variation
        noise = np.random.normal(0, self.noise_level)
        periodic = 0.02 * np.sin(2 * np.pi * current_time / 5)  # 5-second cycle
        thermal_noise = 0.01 * np.sin(2 * np.pi * current_time / 30)  # 30-second thermal cycle
        
        # Calculate new temperature with thermal inertia
        dt = current_time - self.last_read_time
        alpha = 1 - np.exp(-dt / self.response_time)
        target_temp = self.base_temp + drift + noise + periodic + thermal_noise
        new_temp = self.last_reading + alpha * (target_temp - self.last_reading)
        
        self.last_reading = new_temp
        self.last_read_time = current_time
        
        return new_temp
    
    def reset(self):
        """Reset the sensor from a failed state."""
        self.failed = False
        self.last_reading = self.base_temp
        self.last_read_time = 0.0
        self.start_time = 0.0 