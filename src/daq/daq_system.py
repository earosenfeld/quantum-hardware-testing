import time
import random
import numpy as np
from typing import List
from ..cryocooler.sensor import Sensor

class DAQ:
    """Data Acquisition system simulator with realistic network behavior."""
    
    def __init__(self, sensors: List[Sensor], network_latency=0.1, packet_loss_rate=0.01):
        """
        Initialize the DAQ system.
        
        Args:
            sensors (List[Sensor]): List of temperature sensors
            network_latency (float): Simulated network latency in seconds
            packet_loss_rate (float): Probability of packet loss
        """
        self.sensors = sensors
        self.network_latency = network_latency
        self.packet_loss_rate = packet_loss_rate
        self.connected = True
        self.last_read_time = 0.0
        
    def read_all(self, current_time=None) -> List[float]:
        """
        Read all sensors with simulated network effects.
        
        Args:
            current_time (float): Current simulation time in seconds
            
        Returns:
            List[float]: List of temperature readings
            
        Raises:
            ConnectionError: If DAQ is disconnected or experiences network issues
        """
        if current_time is None:
            current_time = time.time()
            
        if not self.connected:
            raise ConnectionError("DAQ is disconnected")
            
        # Simulate network latency
        actual_latency = np.random.normal(self.network_latency, self.network_latency * 0.1)
        # No need to sleep in simulation mode
            
        # Simulate packet loss
        if random.random() < self.packet_loss_rate:
            raise ConnectionError("Network packet loss")
            
        readings = []
        for sensor in self.sensors:
            try:
                reading = sensor.read(current_time)
                readings.append(reading)
            except RuntimeError as e:
                # Log sensor failure and try to recover
                print(f"Sensor error: {e}")
                sensor.reset()
                readings.append(None)
                
        self.last_read_time = current_time
        return [r for r in readings if r is not None]
    
    def reconnect(self):
        """Attempt to reconnect the DAQ system."""
        # No need to sleep in simulation mode
        self.connected = True
        print("DAQ reconnected successfully") 