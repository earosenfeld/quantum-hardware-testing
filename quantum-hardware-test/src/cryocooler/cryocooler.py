import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from .sensor import Sensor
from ..daq.daq_system import DAQ
from .pid import PID
from ..utils.report import generate_csv_report, generate_pdf_report
from ..utils.logger import simple_logger

class CryocoolerTest:
    def __init__(self, test_duration=3600, setpoint=4.0):
        self.test_duration = test_duration
        self.setpoint = setpoint
        self.sensors = [Sensor(i, base_temp=setpoint) for i in range(4)]
        self.daq = DAQ(self.sensors, network_latency=0.1)
        self.pid = PID(
            kp=5.0,
            ki=0.2,
            kd=0.1,
            setpoint=setpoint,
            output_limits=(0, 20),
            sample_time=0.1,
            anti_windup=True
        )
        self.log_entries = []
        self.test_start_time = None
        self.test_results = {
            'stability': None,
            'cooling_rate': None,
            'overshoot': None,
            'settling_time': None,
            'temperature_variance': None
        }

    def calculate_metrics(self, times, temps, pid_outs):
        # Calculate stability (standard deviation of temperature)
        self.test_results['stability'] = np.std(temps[-100:])  # Last 100 readings
        
        # Calculate cooling rate (K/min)
        if len(temps) > 1:
            cooling_rate = (temps[0] - temps[-1]) / ((times[-1] - times[0]) / 60)
            self.test_results['cooling_rate'] = cooling_rate
        
        # Calculate overshoot
        max_temp = max(temps)
        self.test_results['overshoot'] = max_temp - self.setpoint
        
        # Calculate settling time (time to reach within 5% of setpoint)
        settling_threshold = self.setpoint * 0.05
        for i, temp in enumerate(temps):
            if abs(temp - self.setpoint) <= settling_threshold:
                self.test_results['settling_time'] = times[i]
                break
        
        # Calculate temperature variance
        self.test_results['temperature_variance'] = np.var(temps)

    def run_test_cycle(self, plot=True):
        self.test_start_time = 0.0  # Start at 0 for simulated time
        times, temps, pid_outs = [], [], []
        sim_time = 0.0
        time_step = 0.2  # Simulation time step
        
        while sim_time < self.test_duration:
            try:
                temp_readings = self.daq.read_all(current_time=sim_time)
                avg_temp = sum(temp_readings) / len(temp_readings)
                pid_out = self.pid.update(avg_temp, current_time=sim_time)
                
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'avg_temp': avg_temp,
                    'pid_output': pid_out,
                    'sensor_data': temp_readings,
                    'elapsed_time': sim_time
                }
                self.log_entries.append(entry)
                
                times.append(sim_time)
                temps.append(avg_temp)
                pid_outs.append(pid_out)
                
                simple_logger(f"Time: {sim_time:.1f}s, Temp: {avg_temp:.3f} K, PID: {pid_out:.3f}")
                
            except Exception as e:
                simple_logger(f"Error: {e}")
                self.daq.reconnect()
            
            sim_time += time_step
        
        self.calculate_metrics(times, temps, pid_outs)
        
        if plot:
            self.plot_results(times, temps, pid_outs)
        
        return self.log_entries

    def plot_results(self, times, temps, pid_outs):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Temperature plot
        ax1.plot(times, temps, label='Average Temperature (K)')
        ax1.axhline(y=self.setpoint, color='r', linestyle='--', label='Setpoint')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Cryocooler Temperature Profile')
        ax1.legend()
        ax1.grid(True)
        
        # PID output plot
        ax2.plot(times, pid_outs, label='PID Output')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('PID Output')
        ax2.set_title('PID Control Signal')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('reports/temperature_profile.png')
        plt.close()

    def generate_report(self):
        # Create reports directory if it doesn't exist
        Path('reports').mkdir(exist_ok=True)
        
        # Generate CSV report
        df = pd.DataFrame(self.log_entries)
        df.to_csv('reports/test_data.csv', index=False)
        
        # Generate PDF report with test results
        generate_pdf_report(
            test_results=self.test_results,
            test_duration=self.test_duration,
            setpoint=self.setpoint,
            output_file='reports/test_report.pdf'
        )

def main():
    parser = argparse.ArgumentParser(description='Cryocooler Test System')
    parser.add_argument('--run-test', action='store_true', help='Run the test cycle')
    parser.add_argument('--duration', type=int, default=3600, help='Test duration in seconds')
    parser.add_argument('--setpoint', type=float, default=4.0, help='Temperature setpoint in Kelvin')
    parser.add_argument('--generate-report', action='store_true', help='Generate test report')
    
    args = parser.parse_args()
    
    if args.run_test:
        test = CryocoolerTest(test_duration=args.duration, setpoint=args.setpoint)
        test.run_test_cycle()
        
        if args.generate_report:
            test.generate_report()

if __name__ == '__main__':
    main()
