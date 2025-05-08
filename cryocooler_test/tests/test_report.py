import pytest
import os
from src.cryo_cooling_test.report import generate_csv_report, generate_pdf_report

def test_generate_csv_and_pdf_report():
    log_entries = [
        {'timestamp': 1, 'avg_temp': 4.01, 'pid_output': 0.5, 'sensor_data': [4.01, 4.00]},
        {'timestamp': 2, 'avg_temp': 3.99, 'pid_output': 0.6, 'sensor_data': [3.98, 4.00]},
    ]
    csv_file = 'data/test_log.csv'
    pdf_file = 'data/test_report.pdf'
    generate_csv_report(log_entries, csv_file)
    generate_pdf_report(log_entries, pdf_file)
    assert os.path.exists(csv_file)
    assert os.path.exists(pdf_file)
