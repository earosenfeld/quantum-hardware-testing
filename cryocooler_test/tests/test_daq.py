import pytest
import time
from cryo_cooling_test.sensor import Sensor
from cryo_cooling_test.daq import DAQ

def test_daq_data_acquisition():
    sensors = [Sensor(i) for i in range(2)]
    daq = DAQ(sensors)
    data = daq.read_all()
    assert len(data) == 2
    assert all(isinstance(t, float) for t in data)

def test_daq_network_delay_and_recovery():
    sensors = [Sensor(i) for i in range(2)]
    daq = DAQ(sensors, network_latency=0.5, fail_rate=0.5)
    # Simulate network delay
    t0 = time.time()
    try:
        daq.read_all()
    except Exception:
        daq.reconnect()
        daq.fail_rate = 0.0  # Reset fail rate after reconnect
    t1 = time.time()
    assert t1 - t0 >= 0.5
    # After reconnect, should work
    data = daq.read_all()
    assert len(data) == 2
