# Usage Guide

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### Running Tests

```bash
pytest tests/
```

### Generating Reports

The system can generate both CSV and PDF reports:

```python
from cryo_cooling_test.report import generate_csv_report, generate_pdf_report

# Generate CSV report
generate_csv_report(log_entries, filename='data/processed/test_log.csv')

# Generate PDF report
generate_pdf_report(log_entries, filename='reports/test_report.pdf')
```

### Data Collection

Data is automatically stored in the `data/raw` directory during operation.
Processed data and analysis results are stored in `data/processed`.

## Configuration

Configuration files should be placed in the project root directory.
See individual module documentation for specific configuration options. 