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
    # Create a sensor with deterministic failure
    sensor = Sensor(0)
    sensor.failed = True  # Force the failure state
    sensors = [sensor, Sensor(1)]
    daq = DAQ(sensors)
    
    # First read should fail
    with pytest.raises(RuntimeError):
        daq.read_all()
    
    # Reset the sensor
    sensor.reset()
    
    # Now it should work
    data = daq.read_all()
    assert len(data) == 2
