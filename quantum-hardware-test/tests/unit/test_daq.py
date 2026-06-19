import pytest
from src.cryocooler.sensor import Sensor
from src.daq.daq_system import DAQ

def test_daq_data_acquisition():
    sensors = [Sensor(i) for i in range(2)]
    daq = DAQ(sensors)
    data = daq.read_all()
    assert len(data) == 2
    assert all(isinstance(t, float) for t in data)

def test_daq_disconnect_and_recovery():
    """A disconnected DAQ raises ConnectionError until reconnected, then resumes."""
    sensors = [Sensor(i, fail_rate=0.0) for i in range(2)]
    daq = DAQ(sensors, packet_loss_rate=0.0)

    daq.connected = False
    with pytest.raises(ConnectionError):
        daq.read_all(current_time=0.0)

    daq.reconnect()
    data = daq.read_all(current_time=0.1)
    assert len(data) == 2
    assert all(isinstance(t, float) for t in data)
