import pytest
import os
import pandas as pd
from datetime import datetime
from src.cryocooler.sensor import Sensor
from src.daq.daq_system import DAQ
from src.utils.report import generate_csv_report


def test_data_logging_and_csv():
    sensors = [Sensor(i) for i in range(2)]
    daq = DAQ(sensors)
    log_entries = []
    for _ in range(5):
        data = daq.read_all()
        log_entries.append({'timestamp': _, 'sensor_data': data})
    filename = 'data/test_log.csv'
    generate_csv_report(log_entries, filename)
    assert os.path.exists(filename)
    df = pd.read_csv(filename)
    assert len(df) == 5

def test_sensor_failure_handling():
    """DAQ recovers from a failed sensor: it drops + auto-resets the bad channel
    and keeps serving healthy readings, with the sensor back online next cycle."""
    s0 = Sensor(0, fail_rate=0.0)
    s1 = Sensor(1, fail_rate=0.0)
    s0.failed = True  # force sensor 0 into a failed state
    daq = DAQ([s0, s1], packet_loss_rate=0.0)

    # Failed channel is dropped (and auto-reset) this cycle; healthy one served.
    data = daq.read_all(current_time=0.0)
    assert len(data) == 1

    # After the DAQ's auto-reset, both sensors read on the next cycle.
    data = daq.read_all(current_time=0.1)
    assert len(data) == 2
