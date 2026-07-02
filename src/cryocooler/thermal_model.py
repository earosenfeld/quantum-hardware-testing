import numpy as np

class ThermalModel:
    """Simulates thermal behavior of a cryogenic system."""
    
    def __init__(self, initial_temp=4.0, thermal_mass=1000.0, cooling_power=5.0,
                 ambient_temp=300.0, heat_leak_coefficient=0.001):
        """
        Initialize thermal model.
        
        Args:
            initial_temp (float): Initial temperature in Kelvin
            thermal_mass (float): System thermal mass in J/K
            cooling_power (float): Base cooling power in Watts
            ambient_temp (float): Ambient temperature in Kelvin
            heat_leak_coefficient (float): Heat leak coefficient in W/K
        """
        self.temperature = initial_temp
        self.thermal_mass = thermal_mass
        self.cooling_power = cooling_power
        self.ambient_temp = ambient_temp
        self.heat_leak_coefficient = heat_leak_coefficient
        self.cooling_power_multiplier = 1.0  # Controlled by PID
        self.min_temp = 2.0  # Minimum achievable temperature
        self.max_temp = 300.0  # Maximum temperature (ambient)
        
    def update(self, dt):
        """
        Update thermal state for one time step.
        
        Args:
            dt (float): Time step in seconds
            
        Returns:
            float: New temperature in Kelvin
        """
        # Temperature-dependent heat transfer coefficient
        # Increases with temperature difference to model convection
        temp_diff = self.ambient_temp - self.temperature
        heat_transfer_coef = self.heat_leak_coefficient * (1 + 0.01 * abs(temp_diff))
        
        # Calculate heat flows
        heat_leak = heat_transfer_coef * temp_diff
        
        # Temperature-dependent cooling efficiency
        # Cooling power decreases as temperature approaches minimum
        cooling_efficiency = 1.0 - (self.temperature - self.min_temp) / (self.ambient_temp - self.min_temp)
        cooling = self.cooling_power * self.cooling_power_multiplier * max(0, cooling_efficiency)
        
        # Calculate temperature change with stability limits
        dT = (heat_leak - cooling) * dt / self.thermal_mass
        
        # Limit temperature change for stability
        max_dT = 0.1  # Maximum temperature change per step
        dT = np.clip(dT, -max_dT, max_dT)
        
        # Update temperature with bounds
        self.temperature = np.clip(self.temperature + dT, self.min_temp, self.max_temp)
        
        return self.temperature
    
    def set_cooling_power(self, multiplier):
        """
        Set cooling power multiplier (controlled by PID).
        
        Args:
            multiplier (float): Cooling power multiplier (0 to 2)
        """
        # Smooth control changes to prevent rapid oscillations
        max_change = 0.1
        target = np.clip(multiplier, 0, 2)
        self.cooling_power_multiplier = np.clip(
            self.cooling_power_multiplier + np.clip(
                target - self.cooling_power_multiplier,
                -max_change,
                max_change
            ),
            0,
            2
        ) 