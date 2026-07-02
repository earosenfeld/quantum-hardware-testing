"""Main entry point for the cryocooler test package."""
import argparse
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.system.test_scenarios import run_all_scenarios

def main():
    parser = argparse.ArgumentParser(description='Cryocooler Test System')
    parser.add_argument('--run-scenarios', action='store_true',
                      help='Run all test scenarios and generate reports')
    parser.add_argument('--run-test', action='store_true',
                      help='Run the standard test cycle')
    parser.add_argument('--duration', type=int, default=3600,
                      help='Test duration in seconds')
    parser.add_argument('--setpoint', type=float, default=4.0,
                      help='Temperature setpoint in Kelvin')
    parser.add_argument('--generate-report', action='store_true',
                      help='Generate test report')
    
    args = parser.parse_args()
    
    if args.run_scenarios:
        run_all_scenarios()
    elif args.run_test:
        from .cryocooler import CryocoolerTest
        test = CryocoolerTest(test_duration=args.duration, setpoint=args.setpoint)
        test.run_test_cycle()
        
        if args.generate_report:
            test.generate_report()
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 