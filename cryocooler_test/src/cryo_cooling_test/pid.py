class PID:
    def __init__(self, kp, ki, kd, setpoint, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits if output_limits else (-float('inf'), float('inf'))
        
        self.reset()
        
    def reset(self):
        """Reset the PID controller."""
        self.last_error = 0
        self.integral = 0
        self.last_time = None
        self.last_output = 0
        self.last_input = None
        
    def update(self, current_value, current_time):
        """Update the PID controller."""
        error = self.setpoint - current_value
        
        # Handle first update
        if self.last_time is None:
            self.last_time = current_time
            self.last_error = error
            self.last_input = current_value
            # Initial output proportional to error
            output = self.kp * error
            self.last_output = self._clamp(output)
            return self.last_output
        
        # Time difference
        dt = current_time - self.last_time
        if dt <= 0:
            return self.last_output
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        if abs(error) > 0.05:  # Only integrate when error is significant
            self.integral += error * dt
            # Limit integral term
            self.integral = max(min(self.integral, 2.0/self.ki), -2.0/self.ki)
        else:
            self.integral *= 0.95  # Decay integral when close to setpoint
        i_term = self.ki * self.integral
        
        # Derivative term on measurement with filtering
        d_input = current_value - self.last_input
        alpha = 0.1  # Derivative filter coefficient
        d_term = -self.kd * (d_input / dt) * alpha if dt > 0 else 0
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Rate limiting to prevent overshoot
        max_change = 0.5  # Fixed rate limiting
        if abs(output - self.last_output) > max_change:
            if output > self.last_output:
                output = self.last_output + max_change
            else:
                output = self.last_output - max_change
        
        # Apply output limits
        output = self._clamp(output)
        
        # Store values for next iteration
        self.last_error = error
        self.last_time = current_time
        self.last_output = output
        self.last_input = current_value
        
        return output
    
    def _clamp(self, value):
        """Clamp value between output limits."""
        return max(min(value, self.output_limits[1]), self.output_limits[0]) 