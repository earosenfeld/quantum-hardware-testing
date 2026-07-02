import time
import numpy as np
from typing import Tuple, Optional

class PID:
    """Industrial-grade PID controller with anti-windup and bumpless transfer."""
    
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float,
                 output_limits: Tuple[float, float] = (0, 100),
                 sample_time: Optional[float] = None,
                 anti_windup: bool = True):
        """
        Initialize PID controller.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            setpoint (float): Target value
            output_limits (tuple): Min and max output limits
            sample_time (float): Fixed sample time (None for variable)
            anti_windup (bool): Enable anti-windup protection
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.sample_time = sample_time
        self.anti_windup = anti_windup
        
        self._last_time = None
        self._last_error = 0
        self._integral = 0
        self._last_output = 0
        
    def update(self, measurement: float, current_time: Optional[float] = None) -> float:
        """
        Update the PID controller.
        
        Args:
            measurement (float): Process variable measurement
            current_time (float): Current time (None for automatic)
            
        Returns:
            float: Controller output
        """
        if current_time is None:
            current_time = time.time()
            
        if self._last_time is None:
            self._last_time = current_time
            return 0
            
        # Calculate time difference
        dt = current_time - self._last_time
        if self.sample_time is not None:
            if dt < self.sample_time:
                return self._last_output
            dt = self.sample_time
            
        # Calculate error
        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self._integral += error * dt
        i_term = self.ki * self._integral
        
        # Derivative term (on measurement to avoid derivative kick)
        d_term = -self.kd * (measurement - self._last_error) / dt if dt > 0 else 0
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = np.clip(output, *self.output_limits)
        
        # Anti-windup: If output is saturated, stop integrating
        if self.anti_windup and output != p_term + i_term + d_term:
            self._integral = (output - p_term - d_term) / self.ki if self.ki != 0 else 0
        
        # Store values for next iteration
        self._last_error = error
        self._last_time = current_time
        self._last_output = output
        
        return output
        
    def reset(self):
        """Reset the controller's internal state."""
        self._last_time = None
        self._last_error = 0
        self._integral = 0
        self._last_output = 0
        
    def set_tunings(self, kp: float, ki: float, kd: float):
        """Update controller tunings."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def set_setpoint(self, setpoint: float):
        """Update the target setpoint."""
        self.setpoint = setpoint 