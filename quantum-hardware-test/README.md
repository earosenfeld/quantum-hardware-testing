# Cryocooler Test System

A comprehensive testing and control system for cryocooling systems, featuring temperature control, data acquisition, and automated reporting.

## Project Structure

```
project_root/
│
├── src/                      # Source code (core implementation)
│   ├── cryocooler/           # Cryocooler testing logic and components
│   ├── daq/                  # Data acquisition (DAQ) system components
│   ├── power_supply/         # Power supply testing modules
│   ├── system_integration/   # System integration and feedback control
│   └── utils/                # Utility functions
│
├── tests/                    # All test files
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── system/               # System tests
│   └── fixtures/             # Test fixtures
│
├── data/                     # Test data
│   ├── raw/                  # Raw test data
│   ├── processed/            # Processed data
│   └── manifests/            # Test configuration files
│
├── reports/                  # Test result reports
└── docs/                     # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cryocooler_test
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

Run the cryocooler test system:
```bash
cryocooler-test --run-test --generate-report
```

Options:
- `--run-test`: Run the test cycle
- `--duration`: Test duration in seconds (default: 3600)
- `--setpoint`: Temperature setpoint in Kelvin (default: 4.0)
- `--generate-report`: Generate test report

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

## License

MIT License
