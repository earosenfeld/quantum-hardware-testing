# Cryocooler Test System Simulation

This repository simulates and tests a cryogenic cooling system for quantum computing environments. All hardware and protocols are simulated for robust, automated testing.

## Features
- Simulated temperature sensors and DAQ (Modbus TCP/IP)
- PID control logic
- Real-time monitoring and visualization (matplotlib/plotly)
- Data logging and automated report generation
- Protocol and error handling simulation
- Fully automated, pytest-based test suite

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   pytest
   ```
3. Generate a sample report:
   ```bash
   python -m src.control_computer --run-test --generate-report
   ```

## Visualization
- Real-time plots use either matplotlib or plotly (configurable in code).

## Directory Structure
- `src/` - Main simulation and control modules
- `tests/` - Automated test cases
- `data/` - Output logs and reports
