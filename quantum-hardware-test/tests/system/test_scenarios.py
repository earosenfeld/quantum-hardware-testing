import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.cryocooler.sensor import Sensor
from src.cryocooler.pid import PID
from src.daq.daq_system import DAQ
from src.utils.report import generate_csv_report, generate_pdf_report
from src.cryocooler.thermal_model import ThermalModel
import numpy as np

class TestScenario:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.results = []
        self.start_time = None
        self.end_time = None

    def run(self):
        self.start_time = datetime.now()
        # Test implementation will be added by subclasses
        self.end_time = datetime.now()

    def generate_report(self):
        report_data = {
            'scenario_name': self.name,
            'description': self.description,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': (self.end_time - self.start_time).total_seconds(),
            'results': self.results
        }
        return report_data

class SensorTestScenario(TestScenario):
    def __init__(self):
        super().__init__(
            "Sensor Performance Test",
            "Tests sensor accuracy, stability, and failure recovery"
        )

    def run(self):
        super().run()
        sensors = [Sensor(i, base_temp=4.0, fail_rate=0.0001) for i in range(4)]
        
        # Test 1: Basic Reading Accuracy
        readings = []
        for t in range(100):
            readings.extend([s.read(current_time=t * 0.1) for s in sensors])
        avg_temp = sum(readings) / len(readings)
        self.results.append({
            'test': 'Basic Reading Accuracy',
            'result': 'PASS' if 3.9 <= avg_temp <= 4.1 else 'FAIL',
            'details': f'Average temperature: {avg_temp:.3f}K'
        })

        # Test 2: Failure Recovery
        sensor = sensors[0]
        sensor.failed = True
        try:
            sensor.read(current_time=10.0)
            self.results.append({
                'test': 'Failure Detection',
                'result': 'FAIL',
                'details': 'Failed to detect sensor failure'
            })
        except RuntimeError:
            self.results.append({
                'test': 'Failure Detection',
                'result': 'PASS',
                'details': 'Successfully detected sensor failure'
            })

        sensor.reset()
        try:
            temp = sensor.read(current_time=11.0)
            self.results.append({
                'test': 'Failure Recovery',
                'result': 'PASS',
                'details': f'Successfully recovered, reading: {temp:.3f}K'
            })
        except RuntimeError:
            self.results.append({
                'test': 'Failure Recovery',
                'result': 'FAIL',
                'details': 'Failed to recover from failure'
            })

class DAQTestScenario(TestScenario):
    def __init__(self):
        super().__init__(
            "DAQ System Test",
            "Tests DAQ system reliability, network handling, and data acquisition"
        )

    def run(self):
        super().run()
        sensors = [Sensor(i, base_temp=4.0, fail_rate=0.0001) for i in range(4)]
        daq = DAQ(sensors, network_latency=0.1, packet_loss_rate=0.1)

        # Test 1: Basic Data Acquisition
        try:
            readings = daq.read_all(current_time=0.0)
            self.results.append({
                'test': 'Basic Data Acquisition',
                'result': 'PASS',
                'details': f'Successfully read {len(readings)} sensors'
            })
        except Exception as e:
            self.results.append({
                'test': 'Basic Data Acquisition',
                'result': 'FAIL',
                'details': f'Failed to read sensors: {str(e)}'
            })

        # Test 2: Network Recovery
        daq.connected = False
        try:
            daq.read_all(current_time=1.0)
            self.results.append({
                'test': 'Network Failure Detection',
                'result': 'FAIL',
                'details': 'Failed to detect network disconnection'
            })
        except ConnectionError:
            self.results.append({
                'test': 'Network Failure Detection',
                'result': 'PASS',
                'details': 'Successfully detected network disconnection'
            })

        daq.reconnect()
        try:
            readings = daq.read_all(current_time=2.0)
            self.results.append({
                'test': 'Network Recovery',
                'result': 'PASS',
                'details': f'Successfully recovered, read {len(readings)} sensors'
            })
        except Exception as e:
            self.results.append({
                'test': 'Network Recovery',
                'result': 'FAIL',
                'details': f'Failed to recover: {str(e)}'
            })

class PIDTestScenario(TestScenario):
    def __init__(self):
        super().__init__(
            "PID Controller Test",
            "Tests PID controller performance, stability, and response"
        )
        self.step_response_data = []
        self.disturbance_data = []

    def get_pid_gains(self, temperature):
        """Get temperature-dependent PID gains."""
        # More aggressive gains at higher temperatures
        temp_diff = abs(temperature - 4.0)
        if temp_diff > 1.0:
            return 1.0, 0.2, 0.3  # Higher gains for large deviations
        elif temp_diff > 0.5:
            return 0.7, 0.1, 0.2  # Medium gains for moderate deviations
        else:
            return 0.5, 0.05, 0.1  # Conservative gains near setpoint

    def run(self):
        super().run()
        # Create thermal model and sensor with adjusted parameters
        thermal_model = ThermalModel(
            initial_temp=4.2,
            thermal_mass=1000.0,  # Much larger thermal mass
            cooling_power=5.0,
            heat_leak_coefficient=0.001
        )
        sensor = Sensor(id=0, base_temp=4.2, response_time=0.5, noise_level=0.005, fail_rate=0)
        
        # Test 1: Step Response
        outputs = []
        temperatures = []
        dt = 0.1  # Time step in seconds
        
        for t in range(2000):  # 200 seconds simulation
            # Update thermal model and get temperature reading
            temp = thermal_model.update(dt)
            sensor.base_temp = temp
            current_temp = sensor.read(current_time=t * dt)
            temperatures.append(current_temp)
            
            # Get temperature-dependent PID gains
            kp, ki, kd = self.get_pid_gains(current_temp)
            pid = PID(kp=kp, ki=ki, kd=kd, setpoint=4.0, output_limits=(0, 2))
            
            # Update PID and apply control
            out = pid.update(current_temp, current_time=t * dt)
            outputs.append(out)
            thermal_model.set_cooling_power(out)

        # Store step response data for plotting
        self.step_response_data = temperatures.copy()

        # Calculate final value using weighted average of last 10 seconds
        weights = np.exp(np.linspace(-1, 0, 100))  # Exponential weights
        final_value = np.average(temperatures[-100:], weights=weights)
        
        self.results.append({
            'test': 'Step Response',
            'result': 'PASS' if abs(final_value - 4.0) < 0.1 else 'FAIL',
            'details': f'Final value: {final_value:.3f}K'
        })

        # Test 2: Disturbance Rejection
        thermal_model = ThermalModel(
            initial_temp=4.0,
            thermal_mass=1000.0,
            cooling_power=5.0,
            heat_leak_coefficient=0.001
        )
        sensor.base_temp = 4.0
        sensor.last_reading = 4.0
        temperatures = []
        
        for t in range(2000):  # 200 seconds simulation
            # Add gradual disturbance
            if t == 1000:  # Start disturbance at 100s
                thermal_model.heat_leak_coefficient *= 2  # Smaller disturbance
            
            # Update thermal model and get temperature reading
            temp = thermal_model.update(dt)
            sensor.base_temp = temp
            current_temp = sensor.read(current_time=t * dt)
            temperatures.append(current_temp)
            
            # Get temperature-dependent PID gains
            kp, ki, kd = self.get_pid_gains(current_temp)
            pid = PID(kp=kp, ki=ki, kd=kd, setpoint=4.0, output_limits=(0, 2))
            
            # Update PID and apply control
            out = pid.update(current_temp, current_time=t * dt)
            thermal_model.set_cooling_power(out)

        # Store disturbance rejection data for plotting
        self.disturbance_data = temperatures.copy()

        # Calculate final value using weighted average
        final_value = np.average(temperatures[-100:], weights=weights)
        
        self.results.append({
            'test': 'Disturbance Rejection',
            'result': 'PASS' if abs(final_value - 4.0) < 0.1 else 'FAIL',
            'details': f'Final value after disturbance: {final_value:.3f}K'
        })

    def generate_report(self):
        report_data = super().generate_report()
        report_data['data'] = {
            'step_response': self.step_response_data,
            'disturbance_rejection': self.disturbance_data
        }
        return report_data

def run_all_scenarios():
    scenarios = [
        SensorTestScenario(),
        DAQTestScenario(),
        PIDTestScenario()
    ]

    all_results = []
    for scenario in scenarios:
        scenario.run()
        all_results.append(scenario.generate_report())

    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)

    # Generate CSV report
    df = pd.DataFrame([
        {
            'scenario': r['scenario_name'],
            'test': t['test'],
            'result': t['result'],
            'details': t['details']
        }
        for r in all_results
        for t in r['results']
    ])
    df.to_csv(f'reports/test_scenarios_{timestamp}.csv', index=False)

    # Generate PDF report
    generate_pdf_report(
        test_results={
            'scenarios': all_results,
            'summary': {
                'total_tests': sum(len(r['results']) for r in all_results),
                'passed_tests': sum(
                    sum(1 for t in r['results'] if t['result'] == 'PASS')
                    for r in all_results
                )
            }
        },
        test_duration=sum(r['duration'] for r in all_results),
        setpoint=4.0,
        output_file=f'reports/test_scenarios_{timestamp}.pdf'
    )

if __name__ == '__main__':
    run_all_scenarios() 