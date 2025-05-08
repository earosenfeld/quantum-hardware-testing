import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from .sensor import Sensor
from .daq import DAQ
from .pid import PID
from .report import generate_csv_report, generate_pdf_report
from .utils import simple_logger

def run_test_cycle(duration=10, plot=True):
    sensors = [Sensor(i, base_temp=4.0) for i in range(4)]
    daq = DAQ(sensors, network_latency=0.1)
    pid = PID(kp=0.5, ki=0.1, kd=0.05, setpoint=4.0, output_limits=(0, 1))
    log_entries = []
    t0 = time.time()
    times, temps, pid_outs = [], [], []
    while time.time() - t0 < duration:
        try:
            temp_readings = daq.read_all()
            avg_temp = sum(temp_readings) / len(temp_readings)
            pid_out = pid.update(avg_temp, current_time=time.time() - t0)
            entry = {
                'timestamp': time.time(),
                'avg_temp': avg_temp,
                'pid_output': pid_out,
                'sensor_data': temp_readings
            }
            log_entries.append(entry)
            times.append(time.time() - t0)
            temps.append(avg_temp)
            pid_outs.append(pid_out)
            simple_logger(f"Temp: {avg_temp:.3f} K, PID: {pid_out:.3f}")
        except Exception as e:
            simple_logger(f"Error: {e}")
            daq.reconnect()
        time.sleep(0.2)
    if plot:
        plt.figure()
        plt.plot(times, temps, label='Avg Temp (K)')
        plt.plot(times, pid_outs, label='PID Output')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.title('Cryocooler Test Cycle')
        plt.show()
    return log_entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-test', action='store_true')
    parser.add_argument('--generate-report', action='store_true')
    args = parser.parse_args()
    if args.run_test:
        log_entries = run_test_cycle(duration=10, plot=True)
        if args.generate_report:
            generate_csv_report(log_entries)
            generate_pdf_report(log_entries)
            print('Reports generated in data/')

if __name__ == '__main__':
    main()
