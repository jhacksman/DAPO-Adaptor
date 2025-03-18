#!/usr/bin/env python3
"""
Run all tests for the verification module.
"""

import unittest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.test_verification import TestVerification
from tests.test_mathematical_validation import TestMathematicalValidation

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestVerification))
    test_suite.addTest(unittest.makeSuite(TestMathematicalValidation))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with non-zero status if tests failed
    sys.exit(not result.wasSuccessful())
