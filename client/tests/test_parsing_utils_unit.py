#!/usr/bin/env python3
"""
Unit tests for parsing_utils.py - ECM and YAFU output parsing.

All parsing functions are pure (no I/O), so these tests run fast and
don't need mocking. Covers the core factor extraction pipeline,
sigma matching, progress tracking, and version detection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.parsing_utils import (
    _extract_factors_with_patterns,
    parse_ecm_output,
    parse_ecm_output_multiple,
    extract_sigma_for_factor,
    parse_yafu_ecm_output,
    parse_yafu_auto_factors,
    parse_yafu_output_with_composites,
    count_ecm_steps_completed,
    count_ecm_curves_completed,
    get_ecm_curve_times,
    get_ecm_progress_estimate,
    get_yafu_curves_progress,
    get_yafu_curves_completed,
    extract_program_version,
)


# ============================================================
# _extract_factors_with_patterns - core extraction pipeline
# ============================================================

class TestExtractFactorsWithPatterns:
    """Test the unified factor extraction pipeline."""

    def test_no_factors(self):
        output = "GMP-ECM 7.0.6\nStep 1 took 100ms\nStep 2 took 200ms\n"
        assert _extract_factors_with_patterns(output) == []

    def test_empty_output(self):
        assert _extract_factors_with_patterns("") == []

    def test_single_prime_factor_announcement(self):
        """Standard CPU ECM: 'Found prime factor of N digits: XXXXX'"""
        output = (
            "Factor found in step 1: 59460190057621\n"
            "Found prime factor of 14 digits: 59460190057621\n"
        )
        factors = _extract_factors_with_patterns(output)
        assert len(factors) == 1
        assert factors[0][0] == "59460190057621"
        assert factors[0][2] == "PRIME_FACTOR"

    def test_standard_factor_without_prime_announcement(self):
        """Factor found but no 'Found prime factor' line - uses STANDARD_FACTOR."""
        output = "Factor found in step 2: 67280421310721\n"
        factors = _extract_factors_with_patterns(output)
        assert len(factors) == 1
        assert factors[0][0] == "67280421310721"

    def test_asterisk_prefix_stage2(self):
        """Stage 2 factors have asterisk prefix."""
        output = "********** Factor found in step 2: 154848006894803752593902015592419621459239\n"
        factors = _extract_factors_with_patterns(output)
        assert len(factors) == 1
        assert factors[0][0] == "154848006894803752593902015592419621459239"

    def test_gpu_factor_with_sigma(self):
        """GPU format includes sigma in the match."""
        output = "GPU: factor 856395168938929 found in Step 1 with curve 5 (-sigma 3:1111111111)\n"
        factors = _extract_factors_with_patterns(output)
        assert len(factors) == 1
        assert factors[0][0] == "856395168938929"
        assert factors[0][1] == "3:1111111111"

    def test_gpu_factor_without_sigma(self):
        """GPU factor line without sigma group."""
        output = "GPU: factor 856395168938929 found in Step 1 with curve 5\n"
        factors = _extract_factors_with_patterns(output)
        assert len(factors) == 1
        assert factors[0][0] == "856395168938929"
        assert factors[0][1] is None

    def test_prime_matched_to_gpu_sigma(self):
        """Prime announcement should get sigma from matching GPU factor."""
        output = (
            "GPU: factor 856395168938929 found in Step 1 with curve 5 (-sigma 3:1111111111)\n"
            "Found prime factor of 15 digits: 856395168938929\n"
        )
        factors = _extract_factors_with_patterns(output)
        assert len(factors) == 1
        assert factors[0][0] == "856395168938929"
        assert factors[0][1] == "3:1111111111"
        assert factors[0][2] == "PRIME_FACTOR"

    def test_composite_filtered_out(self):
        """GPU reports p1*p2 as a factor - should be filtered as composite."""
        p1 = "856395168938929"
        p2 = "901149811757719"
        composite = str(int(p1) * int(p2))

        output = (
            f"GPU: factor {p1} found in Step 1 with curve 5 (-sigma 3:111)\n"
            f"GPU: factor {p2} found in Step 1 with curve 12 (-sigma 3:222)\n"
            f"GPU: factor {composite} found in Step 1 with curve 20 (-sigma 3:333)\n"
            f"Found prime factor of 15 digits: {p1}\n"
            f"Found prime factor of 15 digits: {p2}\n"
        )
        factors = _extract_factors_with_patterns(output)
        factor_values = [f[0] for f in factors]

        assert p1 in factor_values
        assert p2 in factor_values
        assert composite not in factor_values
        assert len(factors) == 2

    def test_multiple_primes_with_correct_sigmas(self):
        """Each prime should get its own sigma from the GPU line that found it."""
        output = (
            "GPU: factor 856395168938929 found in Step 1 with curve 5 (-sigma 3:1111)\n"
            "GPU: factor 901149811757719 found in Step 1 with curve 12 (-sigma 3:2222)\n"
            "Found prime factor of 15 digits: 856395168938929\n"
            "Found prime factor of 15 digits: 901149811757719\n"
        )
        factors = _extract_factors_with_patterns(output)
        sigma_map = {f[0]: f[1] for f in factors}

        assert sigma_map["856395168938929"] == "3:1111"
        assert sigma_map["901149811757719"] == "3:2222"

    def test_prime_divides_composite_gpu_factor(self):
        """Prime from PRIME_FACTOR that divides a larger GPU composite gets that sigma."""
        # Suppose GPU found composite = prime * cofactor
        prime = "7"
        composite_val = 7 * 11  # 77
        output = (
            f"GPU: factor {composite_val} found in Step 1 with curve 3 (-sigma 3:9999)\n"
            f"Found prime factor of 1 digits: {prime}\n"
        )
        factors = _extract_factors_with_patterns(output)
        # prime 7 should match to the composite's sigma via divisibility
        assert len(factors) >= 1
        prime_entry = [f for f in factors if f[0] == prime]
        assert len(prime_entry) == 1
        assert prime_entry[0][1] == "3:9999"

    def test_deduplication(self):
        """Same factor appearing in multiple patterns should only appear once."""
        output = (
            "Factor found in step 1: 59460190057621\n"
            "Found prime factor of 14 digits: 59460190057621\n"
        )
        factors = _extract_factors_with_patterns(output)
        factor_values = [f[0] for f in factors]
        assert factor_values.count("59460190057621") == 1

    def test_standard_factor_gets_sigma_from_output(self):
        """STANDARD_FACTOR should attempt sigma extraction from surrounding output."""
        output = (
            "Using B1=50000, B2=5000000, polynomial x^1, sigma=3:42\n"
            "-sigma 3:42\n"
            "Factor found in step 1: 12345\n"
            "Found prime factor of 5 digits: 12345\n"
        )
        factors = _extract_factors_with_patterns(output)
        assert len(factors) == 1
        # Should find sigma via fallback extraction
        assert factors[0][1] is not None


# ============================================================
# parse_ecm_output - single factor convenience
# ============================================================

class TestParseEcmOutput:

    def test_returns_first_factor(self):
        output = (
            "Found prime factor of 14 digits: 59460190057621\n"
            "Factor found in step 1: 59460190057621\n"
        )
        factor, sigma = parse_ecm_output(output)
        assert factor == "59460190057621"

    def test_no_factor(self):
        output = "Step 1 took 100ms\nStep 2 took 200ms\n"
        factor, sigma = parse_ecm_output(output)
        assert factor is None
        assert sigma is None

    def test_empty_output(self):
        factor, sigma = parse_ecm_output("")
        assert factor is None
        assert sigma is None


# ============================================================
# parse_ecm_output_multiple
# ============================================================

class TestParseEcmOutputMultiple:

    def test_returns_factor_sigma_pairs(self):
        """Use coprime factors (223 doesn't divide 111) to avoid composite filtering."""
        output = (
            "GPU: factor 111 found in Step 1 with curve 1 (-sigma 3:100)\n"
            "GPU: factor 223 found in Step 1 with curve 2 (-sigma 3:200)\n"
        )
        factors = parse_ecm_output_multiple(output)
        assert len(factors) == 2
        # Each tuple should be (factor, sigma) - no pattern name
        assert all(len(f) == 2 for f in factors)

    def test_no_factors(self):
        assert parse_ecm_output_multiple("no factors here") == []


# ============================================================
# extract_sigma_for_factor
# ============================================================

class TestExtractSigmaForFactor:

    def test_gpu_format(self):
        output = "GPU: factor 12345 found in Step 1 with curve 5 (-sigma 3:99999)\n"
        sigma = extract_sigma_for_factor(output, "12345")
        assert sigma == "3:99999"

    def test_backward_search_from_position(self):
        """Look backwards from factor position for 'Using ... sigma=' line."""
        output = (
            "Using B1=50000, B2=5000000, polynomial x^1, sigma=1:42\n"
            "Step 1 took 100ms\n"
            "Factor found in step 1: 12345\n"
        )
        pos = output.index("Factor found")
        sigma = extract_sigma_for_factor(output, "12345", factor_position=pos)
        assert sigma == "1:42"

    def test_global_fallback(self):
        """Fall back to global -sigma parameter search."""
        output = "-sigma 3:777\nsome other output\n"
        sigma = extract_sigma_for_factor(output, "99999")
        assert sigma == "3:777"

    def test_no_sigma_found(self):
        output = "no sigma information here\n"
        sigma = extract_sigma_for_factor(output, "12345")
        assert sigma is None


# ============================================================
# YAFU parsing
# ============================================================

class TestParseYafuEcmOutput:

    def test_factor_section(self):
        output = (
            "some preamble\n"
            "***factors found***\n"
            "P15 = 856395168938929\n"
            "P31 = 1040100281593968479843247348533\n"
        )
        factors = parse_yafu_ecm_output(output)
        assert len(factors) == 2
        assert factors[0][0] == "856395168938929"
        assert factors[1][0] == "1040100281593968479843247348533"
        # YAFU doesn't report sigma
        assert all(f[1] is None for f in factors)

    def test_multiplicity_preserved(self):
        output = (
            "***factors found***\n"
            "P15 = 874392113604259\n"
            "P15 = 874392113604259\n"
            "P15 = 874392113604259\n"
            "P15 = 856395168938929\n"
        )
        factors = parse_yafu_ecm_output(output)
        assert len(factors) == 4
        factor_values = [f[0] for f in factors]
        assert factor_values.count("874392113604259") == 3

    def test_no_factor_section(self):
        output = "no factors at all\n"
        factors = parse_yafu_ecm_output(output)
        assert factors == []

    def test_ignores_factors_before_section(self):
        """Factors reported during execution (before section) should be ignored."""
        output = (
            "ecm: found prp15 factor = 856395168938929\n"
            "***factors found***\n"
            "P15 = 856395168938929\n"
        )
        factors = parse_yafu_ecm_output(output)
        # Should only have one (from the section), not two
        assert len(factors) == 1


class TestParseYafuAutoFactors:

    def test_identical_to_ecm_output(self):
        """parse_yafu_auto_factors should behave the same as parse_yafu_ecm_output."""
        output = (
            "***factors found***\n"
            "P15 = 856395168938929\n"
            "P31 = 1040100281593968479843247348533\n"
        )
        ecm_factors = parse_yafu_ecm_output(output)
        auto_factors = parse_yafu_auto_factors(output)
        assert ecm_factors == auto_factors


class TestParseYafuOutputWithComposites:

    def test_primes_and_composites(self):
        output = (
            "***factors found***\n"
            "P15 = 856395168938929\n"
            "C37 = 3013157030613612614987357947254984059\n"
            "P31 = 1040100281593968479843247348533\n"
        )
        result = parse_yafu_output_with_composites(output)
        assert len(result['primes']) == 2
        assert len(result['composites']) == 1
        assert "856395168938929" in result['primes']
        assert "3013157030613612614987357947254984059" in result['composites']

    def test_primes_only(self):
        output = (
            "***factors found***\n"
            "P15 = 856395168938929\n"
        )
        result = parse_yafu_output_with_composites(output)
        assert len(result['primes']) == 1
        assert len(result['composites']) == 0

    def test_composites_only(self):
        output = (
            "***factors found***\n"
            "C37 = 3013157030613612614987357947254984059\n"
        )
        result = parse_yafu_output_with_composites(output)
        assert len(result['primes']) == 0
        assert len(result['composites']) == 1

    def test_no_factors(self):
        result = parse_yafu_output_with_composites("no factor section\n")
        assert result['primes'] == []
        assert result['composites'] == []


# ============================================================
# Progress tracking
# ============================================================

class TestCountEcmSteps:

    def test_counts_step1(self):
        output = "Step 1 took 100ms\nStep 1 took 200ms\nStep 1 took 150ms\n"
        assert count_ecm_steps_completed(output) == 3

    def test_zero_steps(self):
        assert count_ecm_steps_completed("no steps here") == 0


class TestCountEcmCurves:

    def test_counts_step2(self):
        output = "Step 2 took 100ms\nStep 2 took 200ms\n"
        assert count_ecm_curves_completed(output) == 2

    def test_alt_pattern(self):
        output = "ECM: Step 2 took 100ms\nECM: Step 2 took 200ms\n"
        assert count_ecm_curves_completed(output) == 2

    def test_zero_curves(self):
        assert count_ecm_curves_completed("") == 0


class TestGetEcmCurveTimes:

    def test_extracts_times(self):
        output = "Step 2 took 100ms\nStep 2 took 250ms\nStep 2 took 175ms\n"
        times = get_ecm_curve_times(output)
        assert times == [100, 250, 175]

    def test_no_times(self):
        assert get_ecm_curve_times("nothing here") == []


class TestGetEcmProgressEstimate:

    def test_basic_progress(self):
        output = "Step 1 took 50ms\nStep 2 took 100ms\nStep 1 took 50ms\nStep 2 took 200ms\n"
        progress = get_ecm_progress_estimate(output, target_curves=10)
        assert progress['curves_completed'] == 2
        assert progress['steps_completed'] == 2
        assert progress['progress_percent'] == 20.0
        assert progress['has_factors'] is False

    def test_with_factor(self):
        output = "Found prime factor of 10 digits: 1234567890\n"
        progress = get_ecm_progress_estimate(output)
        assert progress['has_factors'] is True

    def test_no_target(self):
        output = "Step 2 took 100ms\n"
        progress = get_ecm_progress_estimate(output)
        assert 'progress_percent' not in progress


# ============================================================
# YAFU progress
# ============================================================

class TestYafuProgress:

    def test_curve_progress(self):
        output = "curve 5 of 100\ncurve 10 of 100\n"
        current, total = get_yafu_curves_progress(output)
        assert current == 10  # last match
        assert total == 100

    def test_no_progress(self):
        current, total = get_yafu_curves_progress("no progress info")
        assert current is None
        assert total is None

    def test_curves_completed(self):
        output = "completed 50 curves\ncompleted 75 curves\n"
        assert get_yafu_curves_completed(output) == 75  # max

    def test_no_completed(self):
        assert get_yafu_curves_completed("nothing") is None


# ============================================================
# Version extraction
# ============================================================

class TestExtractProgramVersion:

    def test_ecm_version(self):
        output = "GMP-ECM 7.0.6 [configured with GMP 6.3.0, --enable-asm-redc] [ECM]\n"
        assert extract_program_version(output, 'ecm') == "7.0.6"

    def test_ecm_version_two_part(self):
        output = "GMP-ECM 7.0 [configured]\n"
        assert extract_program_version(output, 'ecm') == "7.0"

    def test_ecm_no_version(self):
        assert extract_program_version("no version info", 'ecm') == "unknown"

    def test_yafu_version(self):
        output = "YAFU Version 2.11\n"
        assert extract_program_version(output, 'yafu') == "2.11"

    def test_yafu_no_version(self):
        assert extract_program_version("no version", 'yafu') == "unknown"

    def test_unknown_program(self):
        assert extract_program_version("anything", 'unknown_prog') == "unknown"


# ============================================================
# Realistic multi-line output scenarios
# ============================================================

class TestRealisticOutputs:
    """Integration-style tests with realistic GMP-ECM output snippets."""

    def test_cpu_ecm_factor_in_step1(self):
        output = """GMP-ECM 7.0.6 [configured with GMP 6.3.0, --enable-asm-redc] [ECM]
Input number is 12345678901234567890123456789 (29 digits)
Using B1=250000, B2=128992510, polynomial Dickson(6), sigma=1:3879529323
Step 1 took 234ms
Step 2 took 567ms
********** Factor found in step 2: 67280421310721
Found prime factor of 14 digits: 67280421310721
Composite cofactor 183456789012345 has 15 digits
"""
        factor, sigma = parse_ecm_output(output)
        assert factor == "67280421310721"

    def test_cpu_ecm_no_factor(self):
        output = """GMP-ECM 7.0.6 [configured with GMP 6.3.0] [ECM]
Input number is 12345678901234567890123456789 (29 digits)
Using B1=250000, B2=128992510, polynomial Dickson(6), sigma=1:1234567
Step 1 took 234ms
Step 2 took 567ms
"""
        factor, sigma = parse_ecm_output(output)
        assert factor is None

    def test_gpu_multiple_factors_realistic(self):
        """Realistic GPU output finding two primes and their product."""
        output = """GMP-ECM 7.0.6 [configured with GMP 6.3.0, --enable-asm-redc] [ECM]
Input number is 771740345279535829905655342951 (30 digits)
Using B1=1000000, B2=100000000, polynomial x^1, sigma=3:9999999999

GPU: factor 856395168938929 found in Step 1 with curve 5 (-sigma 3:1111111111)
GPU: factor 901149811757719 found in Step 1 with curve 12 (-sigma 3:2222222222)
GPU: factor 771740345279535829905655342951 found in Step 1 with curve 20 (-sigma 3:1234567890)

Found prime factor of 15 digits: 856395168938929
Found prime factor of 15 digits: 901149811757719
"""
        factors = parse_ecm_output_multiple(output)
        assert len(factors) == 2

        sigma_map = dict(factors)
        assert sigma_map["856395168938929"] == "3:1111111111"
        assert sigma_map["901149811757719"] == "3:2222222222"
        # Composite (product of the two primes) should be filtered
        assert "771740345279535829905655342951" not in sigma_map

    def test_pm1_factor(self):
        """P-1 method output."""
        output = """GMP-ECM 7.0.6 [configured with GMP 6.3.0] [P-1]
Input number is 12345678901 (11 digits)
Using B1=50000, B2=5000000
Step 1 took 5ms
Step 2 took 10ms
********** Factor found in step 2: 857
Found prime factor of 3 digits: 857
"""
        factor, sigma = parse_ecm_output(output)
        assert factor == "857"

    def test_pp1_factor(self):
        """P+1 method output."""
        output = """GMP-ECM 7.0.6 [configured with GMP 6.3.0] [P+1]
Input number is 1234567890123 (13 digits)
Using B1=50000, B2=5000000
Step 1 took 5ms
Step 2 took 10ms
Factor found in step 1: 1234577
Found prime factor of 7 digits: 1234577
"""
        factor, sigma = parse_ecm_output(output)
        assert factor == "1234577"
