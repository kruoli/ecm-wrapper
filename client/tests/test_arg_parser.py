#!/usr/bin/env python3
"""
Unit tests for argument parser utilities.
"""
import unittest
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.arg_parser import parse_int_with_scientific


class TestParseIntWithScientific(unittest.TestCase):
    """Test cases for parse_int_with_scientific function."""

    def test_regular_integers(self):
        """Test parsing regular integer strings."""
        self.assertEqual(parse_int_with_scientific("1000000"), 1000000)
        self.assertEqual(parse_int_with_scientific("42"), 42)
        self.assertEqual(parse_int_with_scientific("0"), 0)
        self.assertEqual(parse_int_with_scientific("999999999"), 999999999)

    def test_scientific_notation_lowercase_e(self):
        """Test parsing scientific notation with lowercase 'e'."""
        self.assertEqual(parse_int_with_scientific("1e6"), 1000000)
        self.assertEqual(parse_int_with_scientific("26e7"), 260000000)
        self.assertEqual(parse_int_with_scientific("4e11"), 400000000000)
        self.assertEqual(parse_int_with_scientific("5e2"), 500)
        self.assertEqual(parse_int_with_scientific("1e0"), 1)

    def test_scientific_notation_uppercase_e(self):
        """Test parsing scientific notation with uppercase 'E'."""
        self.assertEqual(parse_int_with_scientific("1E6"), 1000000)
        self.assertEqual(parse_int_with_scientific("26E7"), 260000000)
        self.assertEqual(parse_int_with_scientific("4E11"), 400000000000)

    def test_scientific_notation_with_decimal(self):
        """Test parsing scientific notation with decimal mantissa."""
        self.assertEqual(parse_int_with_scientific("1.5e6"), 1500000)
        self.assertEqual(parse_int_with_scientific("2.6e7"), 26000000)
        self.assertEqual(parse_int_with_scientific("3.14e2"), 314)
        self.assertEqual(parse_int_with_scientific("9.9e3"), 9900)

    def test_scientific_notation_with_plus_sign(self):
        """Test parsing scientific notation with explicit positive exponent."""
        self.assertEqual(parse_int_with_scientific("1e+6"), 1000000)
        self.assertEqual(parse_int_with_scientific("26e+7"), 260000000)
        self.assertEqual(parse_int_with_scientific("4E+11"), 400000000000)

    def test_large_numbers(self):
        """Test parsing very large numbers in scientific notation."""
        # 10^15
        self.assertEqual(parse_int_with_scientific("1e15"), 1000000000000000)
        # 5 * 10^18
        self.assertEqual(parse_int_with_scientific("5e18"), 5000000000000000000)

    def test_negative_numbers_raise_error(self):
        """Test that negative numbers raise ArgumentTypeError."""
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            parse_int_with_scientific("-100")
        self.assertIn("must be positive", str(cm.exception))

        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            parse_int_with_scientific("-1e6")
        self.assertIn("must be positive", str(cm.exception))

    def test_invalid_formats_raise_error(self):
        """Test that invalid formats raise ArgumentTypeError."""
        # Non-numeric strings
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            parse_int_with_scientific("abc")
        self.assertIn("Invalid integer", str(cm.exception))

        # Empty string
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            parse_int_with_scientific("")
        self.assertIn("Invalid integer", str(cm.exception))

        # Invalid scientific notation
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            parse_int_with_scientific("1e")
        self.assertIn("Invalid integer", str(cm.exception))

        # Double decimal points
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            parse_int_with_scientific("1.2.3")
        self.assertIn("Invalid integer", str(cm.exception))

    def test_float_strings_are_truncated(self):
        """Test that float strings are truncated to integers."""
        self.assertEqual(parse_int_with_scientific("123.456"), 123)
        self.assertEqual(parse_int_with_scientific("999.99"), 999)

    def test_whitespace_handling(self):
        """Test handling of strings with whitespace."""
        # Python's float() handles leading/trailing whitespace
        self.assertEqual(parse_int_with_scientific("  1000  "), 1000)
        self.assertEqual(parse_int_with_scientific(" 1e6 "), 1000000)

    def test_typical_ecm_bounds(self):
        """Test typical ECM B1/B2 bound values."""
        # Common B1 values
        self.assertEqual(parse_int_with_scientific("11e6"), 11000000)
        self.assertEqual(parse_int_with_scientific("43e6"), 43000000)
        self.assertEqual(parse_int_with_scientific("110e6"), 110000000)
        self.assertEqual(parse_int_with_scientific("260e6"), 260000000)

        # Common B2 values
        self.assertEqual(parse_int_with_scientific("1.9e9"), 1900000000)
        self.assertEqual(parse_int_with_scientific("3.5e10"), 35000000000)
        self.assertEqual(parse_int_with_scientific("1.2e11"), 120000000000)

    def test_zero_values(self):
        """Test zero and zero-like values."""
        self.assertEqual(parse_int_with_scientific("0"), 0)
        self.assertEqual(parse_int_with_scientific("0.0"), 0)
        self.assertEqual(parse_int_with_scientific("0e0"), 0)
        self.assertEqual(parse_int_with_scientific("0e10"), 0)


class TestParserIntegration(unittest.TestCase):
    """Integration tests for argparse with scientific notation."""

    def test_ecm_parser_with_scientific_notation(self):
        """Test that ECM parser correctly handles scientific notation."""
        from lib.arg_parser import create_ecm_parser

        parser = create_ecm_parser()

        # Test with scientific notation
        args = parser.parse_args(['--composite', '12345', '--b1', '26e7', '--b2', '4e11'])
        self.assertEqual(args.b1, 260000000)
        self.assertEqual(args.b2, 400000000000)

        # Test with regular integers
        args = parser.parse_args(['--composite', '12345', '--b1', '1000000', '--b2', '5000000'])
        self.assertEqual(args.b1, 1000000)
        self.assertEqual(args.b2, 5000000)

    def test_yafu_parser_with_scientific_notation(self):
        """Test that YAFU parser correctly handles scientific notation."""
        from lib.arg_parser import create_yafu_parser

        parser = create_yafu_parser()

        # Test with scientific notation
        args = parser.parse_args(['--composite', '12345', '--b1', '11e6', '--b2', '1.9e9'])
        self.assertEqual(args.b1, 11000000)
        self.assertEqual(args.b2, 1900000000)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
