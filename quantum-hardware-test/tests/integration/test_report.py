import os
from src.utils.report import generate_csv_report, generate_pdf_report


def test_generate_csv_report():
    os.makedirs('data', exist_ok=True)
    log_entries = [
        {'timestamp': 1, 'avg_temp': 4.01, 'pid_output': 0.5},
        {'timestamp': 2, 'avg_temp': 3.99, 'pid_output': 0.6},
    ]
    csv_file = 'data/test_log.csv'
    generate_csv_report(log_entries, csv_file)
    assert os.path.exists(csv_file)


def test_generate_pdf_report():
    """Exercise the real PDF report API (structured scenario results)."""
    os.makedirs('data', exist_ok=True)
    test_results = {
        'summary': {'total_tests': 2, 'passed_tests': 2},
        'scenarios': [
            {
                'scenario_name': 'Thermal Stability',
                'description': 'Cryostat holds base temperature within tolerance.',
                'results': [
                    {'test': 'Setpoint tracking', 'result': 'PASS', 'details': '4.00 K'},
                    {'test': 'Noise band', 'result': 'PASS', 'details': '< 10 mK rms'},
                ],
            }
        ],
    }
    pdf_file = 'data/test_report.pdf'
    generate_pdf_report(test_results, test_duration=10.0, setpoint=4.0, output_file=pdf_file)
    assert os.path.exists(pdf_file)
