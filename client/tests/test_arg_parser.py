#!/usr/bin/env python3
"""
Tests for arg_parser module
"""
import sys
from pathlib import Path
from argparse import Namespace

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.arg_parser import (
    create_ecm_parser,
    validate_ecm_args,
    get_method_defaults,
    resolve_gpu_settings,
    resolve_worker_count
)


def test_create_ecm_parser():
    """Test ECM parser creation"""
    parser = create_ecm_parser()

    assert parser is not None, "Should create parser"

    # Test parsing basic arguments
    args = parser.parse_args(['--composite', '12345', '--curves', '100'])

    assert args.composite == '12345', "Should parse composite"
    assert args.curves == 100, "Should parse curves"

    print("✓ test_create_ecm_parser passed")


def test_validate_start_tlevel_requires_tlevel():
    """Test that --start-tlevel requires --tlevel"""
    args = Namespace(
        composite='12345',
        tlevel=None,
        start_tlevel=25.0,
        curves=None,
        b1=None,
        b2=None,
        stage2_only=None,
        resume_residues=None,
        multiprocess=False,
        two_stage=False,
        method='ecm',
        gpu=False,
        no_gpu=False,
        save_residues=None
    )

    errors = validate_ecm_args(args)

    assert 'start_tlevel' in errors, "Should have start_tlevel error"
    assert 'requires --tlevel' in errors['start_tlevel'], "Should require --tlevel"

    print("✓ test_validate_start_tlevel_requires_tlevel passed")


def test_validate_start_tlevel_less_than_target():
    """Test that --start-tlevel must be less than --tlevel"""
    args = Namespace(
        composite='12345',
        tlevel=35.0,
        start_tlevel=40.0,  # Greater than tlevel
        curves=None,
        b1=None,
        b2=None,
        stage2_only=None,
        resume_residues=None,
        multiprocess=False,
        two_stage=False,
        method='ecm',
        gpu=False,
        no_gpu=False,
        save_residues=None
    )

    errors = validate_ecm_args(args)

    assert 'start_tlevel' in errors, "Should have start_tlevel error"
    assert 'must be less than' in errors['start_tlevel'], "Should require start < target"

    print("✓ test_validate_start_tlevel_less_than_target passed")


def test_validate_start_tlevel_non_negative():
    """Test that --start-tlevel must be non-negative"""
    args = Namespace(
        composite='12345',
        tlevel=35.0,
        start_tlevel=-5.0,  # Negative
        curves=None,
        b1=None,
        b2=None,
        stage2_only=None,
        resume_residues=None,
        multiprocess=False,
        two_stage=False,
        method='ecm',
        gpu=False,
        no_gpu=False,
        save_residues=None
    )

    errors = validate_ecm_args(args)

    assert 'start_tlevel' in errors, "Should have start_tlevel error"
    assert 'non-negative' in errors['start_tlevel'], "Should require non-negative"

    print("✓ test_validate_start_tlevel_non_negative passed")


def test_validate_start_tlevel_valid():
    """Test valid --start-tlevel configuration"""
    args = Namespace(
        composite='12345',
        tlevel=35.0,
        start_tlevel=25.0,  # Valid: 0 <= 25 < 35
        curves=None,
        b1=None,
        b2=None,
        stage2_only=None,
        resume_residues=None,
        multiprocess=False,
        two_stage=False,
        method='ecm',
        gpu=False,
        no_gpu=False,
        save_residues=None
    )

    errors = validate_ecm_args(args)

    assert 'start_tlevel' not in errors, "Should not have start_tlevel error"

    print("✓ test_validate_start_tlevel_valid passed")


def test_validate_tlevel_conflicts_with_curves():
    """Test that --tlevel and --curves are mutually exclusive"""
    args = Namespace(
        composite='12345',
        tlevel=35.0,
        curves=100,
        start_tlevel=None,
        b1=None,
        b2=None,
        stage2_only=None,
        resume_residues=None,
        multiprocess=False,
        two_stage=False,
        method='ecm',
        gpu=False,
        no_gpu=False,
        save_residues=None
    )

    errors = validate_ecm_args(args)

    assert 'curves' in errors, "Should have curves error"
    assert 'both --tlevel and --curves' in errors['curves'], "Should reject both"

    print("✓ test_validate_tlevel_conflicts_with_curves passed")


def test_validate_gpu_conflicts():
    """Test that --gpu and --no-gpu are mutually exclusive"""
    args = Namespace(
        composite='12345',
        curves=100,
        tlevel=None,
        start_tlevel=None,
        b1=None,
        b2=None,
        stage2_only=None,
        resume_residues=None,
        multiprocess=False,
        two_stage=False,
        method='ecm',
        gpu=True,
        no_gpu=True,  # Both specified
        save_residues=None
    )

    errors = validate_ecm_args(args)

    assert 'gpu' in errors, "Should have GPU error"

    print("✓ test_validate_gpu_conflicts passed")


def test_validate_multiprocess_conflicts_with_two_stage():
    """Test that --multiprocess and --two-stage are mutually exclusive"""
    args = Namespace(
        composite='12345',
        curves=100,
        tlevel=None,
        start_tlevel=None,
        b1=None,
        b2=None,
        stage2_only=None,
        resume_residues=None,
        multiprocess=True,
        two_stage=True,  # Both specified
        method='ecm',
        gpu=False,
        no_gpu=False,
        save_residues=None
    )

    errors = validate_ecm_args(args)

    assert 'mode' in errors, "Should have mode error"
    assert 'both --multiprocess and --two-stage' in errors['mode'], "Should reject both"

    print("✓ test_validate_multiprocess_conflicts_with_two_stage passed")


def test_get_method_defaults():
    """Test getting default B1/B2 values for methods"""
    config = {
        'programs': {
            'gmp_ecm': {
                'default_b1': 50000,
                'default_b2': 5000000,
                'pm1_b1': 100000,
                'pm1_b2': 10000000,
                'pp1_b1': 150000,
                'pp1_b2': 15000000
            }
        }
    }

    # Test ECM defaults
    b1, b2 = get_method_defaults(config, 'ecm')
    assert b1 == 50000, "Should get ECM B1 default"
    assert b2 == 5000000, "Should get ECM B2 default"

    # Test P-1 defaults
    b1, b2 = get_method_defaults(config, 'pm1')
    assert b1 == 100000, "Should get P-1 B1 default"
    assert b2 == 10000000, "Should get P-1 B2 default"

    # Test P+1 defaults
    b1, b2 = get_method_defaults(config, 'pp1')
    assert b1 == 150000, "Should get P+1 B1 default"
    assert b2 == 15000000, "Should get P+1 B2 default"

    print("✓ test_get_method_defaults passed")


def test_resolve_gpu_settings():
    """Test GPU settings resolution"""
    config = {
        'programs': {
            'gmp_ecm': {
                'gpu_enabled': True,
                'gpu_device': 0,
                'gpu_curves': 1024
            }
        }
    }

    # Test with config defaults
    args = Namespace(gpu=False, no_gpu=False, gpu_device=None, gpu_curves=None)
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, config)
    assert use_gpu is True, "Should use config default"
    assert gpu_device == 0, "Should use config GPU device"
    assert gpu_curves == 1024, "Should use config GPU curves"

    # Test with --no-gpu override
    args = Namespace(gpu=False, no_gpu=True, gpu_device=None, gpu_curves=None)
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, config)
    assert use_gpu is False, "Should override to disable GPU"

    # Test with --gpu override
    args = Namespace(gpu=True, no_gpu=False, gpu_device=1, gpu_curves=2048)
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, config)
    assert use_gpu is True, "Should override to enable GPU"
    assert gpu_device == 1, "Should use command line GPU device"
    assert gpu_curves == 2048, "Should use command line GPU curves"

    print("✓ test_resolve_gpu_settings passed")


def test_resolve_worker_count():
    """Test worker count resolution"""
    # Test with explicit worker count
    args = Namespace(multiprocess=True, workers=4)
    worker_count = resolve_worker_count(args)
    assert worker_count == 4, "Should use explicit worker count"

    # Test with auto-detection (workers=0)
    args = Namespace(multiprocess=True, workers=0)
    worker_count = resolve_worker_count(args)
    assert worker_count > 0, "Should auto-detect CPU count"

    # Test without multiprocess
    args = Namespace(multiprocess=False, workers=4)
    worker_count = resolve_worker_count(args)
    assert worker_count == 4, "Should return workers value"

    print("✓ test_resolve_worker_count passed")


def main():
    """Run all tests"""
    print("Running arg_parser tests...\n")

    tests = [
        test_create_ecm_parser,
        test_validate_start_tlevel_requires_tlevel,
        test_validate_start_tlevel_less_than_target,
        test_validate_start_tlevel_non_negative,
        test_validate_start_tlevel_valid,
        test_validate_tlevel_conflicts_with_curves,
        test_validate_gpu_conflicts,
        test_validate_multiprocess_conflicts_with_two_stage,
        test_get_method_defaults,
        test_resolve_gpu_settings,
        test_resolve_worker_count
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
