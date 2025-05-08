import pytest
from src.protocols import ModbusTCP, VISA

def test_modbus_tcp_error_and_recovery():
    modbus = ModbusTCP(fail_rate=1.0)
    with pytest.raises(ConnectionError):
        modbus.send('test')
    modbus.reconnect()
    modbus.fail_rate = 0.0
    assert modbus.send('test') is True

def test_visa_error_and_recovery():
    visa = VISA(fail_rate=1.0)
    with pytest.raises(ConnectionError):
        visa.send('test')
    visa.reconnect()
    visa.fail_rate = 0.0
    assert visa.send('test') is True
