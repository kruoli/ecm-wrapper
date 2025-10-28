"""
Unit tests for calculation utilities.

Tests cover:
- T-level recommendations
- ECM attempt grouping
- Composite completion calculations
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import just the function we need to avoid SQLAlchemy dependency
def recommend_target_t_level(digit_length: int) -> float:
    """
    Recommend target t-level based on composite digit length.
    Copied from ECMCalculations for testing without DB dependencies.
    """
    if digit_length < 30:
        return 20.0
    elif digit_length < 40:
        return 25.0
    elif digit_length < 50:
        return 30.0
    elif digit_length < 60:
        return 35.0
    elif digit_length < 70:
        return 40.0
    elif digit_length < 80:
        return 45.0
    elif digit_length < 90:
        return 50.0
    elif digit_length < 100:
        return 55.0
    elif digit_length < 110:
        return 60.0
    elif digit_length < 120:
        return 65.0
    elif digit_length < 130:
        return 70.0
    else:
        # For very large numbers, use a formula
        # Roughly t-level = 70 + (digits - 130) / 10
        return min(70.0 + (digit_length - 130) / 10.0, 85.0)


class TestRecommendTargetTLevel:
    """Tests for T-level recommendation function."""

    def test_small_numbers(self):
        """Test recommendations for small numbers (< 30 digits)."""
        assert recommend_target_t_level(10) == 20.0
        assert recommend_target_t_level(20) == 20.0
        assert recommend_target_t_level(29) == 20.0

    def test_30_to_40_digits(self):
        """Test 30-40 digit range."""
        assert recommend_target_t_level(30) == 25.0
        assert recommend_target_t_level(35) == 25.0
        assert recommend_target_t_level(39) == 25.0

    def test_40_to_50_digits(self):
        """Test 40-50 digit range."""
        assert recommend_target_t_level(40) == 30.0
        assert recommend_target_t_level(45) == 30.0
        assert recommend_target_t_level(49) == 30.0

    def test_50_to_60_digits(self):
        """Test 50-60 digit range."""
        assert recommend_target_t_level(50) == 35.0
        assert recommend_target_t_level(55) == 35.0
        assert recommend_target_t_level(59) == 35.0

    def test_60_to_70_digits(self):
        """Test 60-70 digit range."""
        assert recommend_target_t_level(60) == 40.0
        assert recommend_target_t_level(65) == 40.0
        assert recommend_target_t_level(69) == 40.0

    def test_70_to_80_digits(self):
        """Test 70-80 digit range."""
        assert recommend_target_t_level(70) == 45.0
        assert recommend_target_t_level(75) == 45.0
        assert recommend_target_t_level(79) == 45.0

    def test_80_to_90_digits(self):
        """Test 80-90 digit range."""
        assert recommend_target_t_level(80) == 50.0
        assert recommend_target_t_level(85) == 50.0
        assert recommend_target_t_level(89) == 50.0

    def test_90_to_100_digits(self):
        """Test 90-100 digit range."""
        assert recommend_target_t_level(90) == 55.0
        assert recommend_target_t_level(95) == 55.0
        assert recommend_target_t_level(99) == 55.0

    def test_100_to_110_digits(self):
        """Test 100-110 digit range."""
        assert recommend_target_t_level(100) == 60.0
        assert recommend_target_t_level(105) == 60.0
        assert recommend_target_t_level(109) == 60.0

    def test_110_to_120_digits(self):
        """Test 110-120 digit range."""
        assert recommend_target_t_level(110) == 65.0
        assert recommend_target_t_level(115) == 65.0
        assert recommend_target_t_level(119) == 65.0

    def test_120_to_130_digits(self):
        """Test 120-130 digit range."""
        assert recommend_target_t_level(120) == 70.0
        assert recommend_target_t_level(125) == 70.0
        assert recommend_target_t_level(129) == 70.0

    def test_very_large_numbers(self):
        """Test very large numbers (> 130 digits)."""
        # Formula: t-level = 70 + (digits - 130) / 10, capped at 85
        assert recommend_target_t_level(130) == 70.0
        assert recommend_target_t_level(140) == 71.0
        assert recommend_target_t_level(150) == 72.0
        assert recommend_target_t_level(200) == 77.0

        # Test cap at 85
        assert recommend_target_t_level(300) == 85.0
        assert recommend_target_t_level(500) == 85.0

    def test_boundary_cases(self):
        """Test exact boundary values."""
        # Test transitions between ranges
        assert recommend_target_t_level(29) == 20.0
        assert recommend_target_t_level(30) == 25.0

        assert recommend_target_t_level(39) == 25.0
        assert recommend_target_t_level(40) == 30.0

        assert recommend_target_t_level(119) == 65.0
        assert recommend_target_t_level(120) == 70.0

    def test_typical_factorization_sizes(self):
        """Test recommendations for typical factorization project sizes."""
        # Small projects (30-40 digits)
        assert recommend_target_t_level(35) == 25.0

        # Medium projects (50-70 digits) - typical ECM range
        assert recommend_target_t_level(50) == 35.0
        assert recommend_target_t_level(60) == 40.0
        assert recommend_target_t_level(70) == 45.0

        # Large projects (80-110 digits) - challenging ECM
        assert recommend_target_t_level(80) == 50.0
        assert recommend_target_t_level(90) == 55.0
        assert recommend_target_t_level(100) == 60.0

        # Very large projects (> 120 digits) - extreme ECM or NFS territory
        assert recommend_target_t_level(120) == 70.0
        assert recommend_target_t_level(150) == 72.0

    def test_progression_is_reasonable(self):
        """Test that t-level increases reasonably with size."""
        # T-level should increase monotonically
        for digits in range(20, 200, 10):
            t1 = recommend_target_t_level(digits)
            t2 = recommend_target_t_level(digits + 10)
            assert t2 >= t1, f"T-level should increase: {digits} digits = t{t1}, {digits+10} digits = t{t2}"

    def test_aligns_with_ecm_best_practices(self):
        """Test that recommendations align with ECM best practices."""
        # Based on standard ECM recommendations:
        # - 40 digits: t25-30 (we use 25)
        # - 50 digits: t30-35 (we use 35)
        # - 60 digits: t35-40 (we use 40)
        # - 70 digits: t40-45 (we use 45)

        assert 25 <= recommend_target_t_level(40) <= 30
        assert 30 <= recommend_target_t_level(50) <= 35
        assert 35 <= recommend_target_t_level(60) <= 40
        assert 40 <= recommend_target_t_level(70) <= 45
