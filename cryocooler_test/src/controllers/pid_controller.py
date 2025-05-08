"""
PID Controller implementation for temperature control.
"""

import time
from typing import Optional, Tuple

class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float,
        output_limits: Tuple[float, float] = (-100, 100),
        sample_time: float = 0.1,
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target value
            output_limits: Tuple of (min, max) output values
            sample_time: Time between updates in seconds
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.sample_time = sample_time
        
        # Controller state
        self._last_time = 0.0
        self._last_error = 0.0
        self._integral = 0.0
        self._last_output = 0.0
        
    def update(self, process_variable: float, current_time: float) -> float:
        """
        Update PID controller and compute output.

        Args:
            process_variable: Current value
            current_time: Current simulation time

        Returns:
            Control output
        """
        dt = current_time - self._last_time
        
        if dt < self.sample_time and self._last_time != 0:
            return self._last_output
            
        error = self.setpoint - process_variable
        
        # Calculate P term
        p_term = self.kp * error
        
        # Calculate I term with anti-windup
        self._integral += error * dt
        i_term = self.ki * self._integral
        
        # Calculate D term
        if dt > 0:
            derivative = (error - self._last_error) / dt
        else:
            derivative = 0
        d_term = self.kd * derivative
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Anti-windup: If output is saturated, stop integrating
        if output != p_term + i_term + d_term:
            self._integral = (output - p_term - d_term) / self.ki if self.ki != 0 else 0
        
        # Update state
        self._last_time = current_time
        self._last_error = error
        self._last_output = output
        
        return output
        
    def reset(self):
        """Reset controller state."""
        self._last_time = 0.0
        self._last_error = 0.0
        self._integral = 0.0
        self._last_output = 0.0 