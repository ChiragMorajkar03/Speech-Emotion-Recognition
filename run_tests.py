#!/usr/bin/env python
"""
Test runner script for Speech Emotion Recognition project.
This script runs unit tests and generates coverage reports.
"""

import os
import sys
import unittest
import coverage

def run_tests_with_coverage():
    """Run tests with coverage reporting"""
    # Start coverage measurement
    cov = coverage.Coverage(
        source=["src"],
        omit=["*/tests/*", "*/venv/*", "*/env/*", "*/__pycache__/*"]
    )
    cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    tests = loader.discover("tests")
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(tests)
    
    # Stop coverage measurement and generate report
    cov.stop()
    
    print("\nCoverage Summary:")
    cov.report()
    
    # Generate HTML report
    html_report_dir = os.path.join("coverage_html")
    if not os.path.exists(html_report_dir):
        os.makedirs(html_report_dir)
    
    cov.html_report(directory=html_report_dir)
    print(f"HTML coverage report generated in {html_report_dir}/index.html")
    
    # Return test result for CI integration
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests_with_coverage())
