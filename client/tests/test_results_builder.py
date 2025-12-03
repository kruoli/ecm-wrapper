"""Unit tests for ResultsBuilder class."""

import pytest
from lib.results_builder import ResultsBuilder, results_for_ecm, results_for_stage1


def test_basic_construction():
    """Test basic builder with required fields."""
    results = ResultsBuilder('123456789', 'ecm').build()

    assert results['composite'] == '123456789'
    assert results['method'] == 'ecm'
    assert results['raw_output'] == ''
    assert results['factors_found'] == []
    assert results['factor_found'] is None
    assert results['curves_completed'] == 0
    assert results['curves_requested'] == 0


def test_fluent_api():
    """Test method chaining."""
    results = (ResultsBuilder('999', 'ecm')
               .with_b1(50000)
               .with_b2(5000000)
               .with_curves(100, 50)
               .with_parametrization(1)
               .build())

    assert results['b1'] == 50000
    assert results['b2'] == 5000000
    assert results['curves_requested'] == 100
    assert results['curves_completed'] == 50
    assert results['parametrization'] == 1


def test_raw_output_accumulation():
    """Test list accumulation and joining."""
    builder = ResultsBuilder('123', 'ecm')
    builder.add_raw_output('Line 1')
    builder.add_raw_output('Line 2')
    builder.add_raw_outputs(['Line 3', 'Line 4'])

    results = builder.build_no_truncate()
    assert results['raw_output'] == 'Line 1\n\nLine 2\n\nLine 3\n\nLine 4'


def test_raw_output_truncation():
    """Test truncation at 10k chars."""
    builder = ResultsBuilder('123', 'ecm')
    builder.add_raw_output('x' * 15000)

    results = builder.build()  # Default 10k truncation
    assert len(results['raw_output']) < 15000
    assert 'truncated' in results['raw_output']
    assert '15000 total chars' in results['raw_output']


def test_raw_output_no_truncation():
    """Test build_no_truncate preserves full output."""
    builder = ResultsBuilder('123', 'ecm')
    builder.add_raw_output('x' * 15000)

    results = builder.build_no_truncate()
    assert len(results['raw_output']) == 15000
    assert 'truncated' not in results['raw_output']


def test_factors_with_sigmas():
    """Test factor handling with sigmas."""
    all_factors = [('123', '12345'), ('456', '67890')]

    results = (ResultsBuilder('999', 'ecm')
               .with_factors(all_factors)
               .build())

    assert results['factor_found'] == '123'
    assert results['factors_found'] == ['123', '456']
    assert results['factor_sigmas'] == {'123': '12345', '456': '67890'}


def test_factors_without_sigmas():
    """Test factor handling when sigma is None."""
    all_factors = [('123', None), ('456', None)]

    results = (ResultsBuilder('999', 'ecm')
               .with_factors(all_factors)
               .build())

    assert results['factor_found'] == '123'
    assert results['factors_found'] == ['123', '456']
    # Should not create factor_sigmas if all sigmas are None
    assert 'factor_sigmas' not in results


def test_single_factor():
    """Test with_single_factor convenience method."""
    results = (ResultsBuilder('999', 'ecm')
               .with_single_factor('12345', '67890')
               .build())

    assert results['factor_found'] == '12345'
    assert results['factors_found'] == ['12345']
    assert results['factor_sigmas'] == {'12345': '67890'}


def test_single_factor_no_sigma():
    """Test with_single_factor without sigma."""
    results = (ResultsBuilder('999', 'ecm')
               .with_single_factor('12345')
               .build())

    assert results['factor_found'] == '12345'
    assert results['factors_found'] == ['12345']
    assert 'factor_sigmas' not in results


def test_stage1_only():
    """Test stage1-only mode (b2=0)."""
    results = results_for_stage1('123', 50000, 100, param=1).build()

    assert results['b1'] == 50000
    assert results['b2'] == 0
    assert results['curves_requested'] == 100
    assert results['parametrization'] == 1


def test_as_stage1_only():
    """Test as_stage1_only sets b2=0."""
    results = (ResultsBuilder('123', 'ecm')
               .as_stage1_only()
               .build())

    assert results['b2'] == 0


def test_multiprocess_mode():
    """Test multiprocess flags."""
    results = (ResultsBuilder('123', 'ecm')
               .as_multiprocess(8)
               .build())

    assert results['multiprocess'] is True
    assert results['workers'] == 8


def test_two_stage_mode():
    """Test two-stage flag."""
    results = (ResultsBuilder('123', 'ecm')
               .as_two_stage()
               .build())

    assert results['two_stage'] is True


def test_stage2_workers():
    """Test stage2_workers flag."""
    results = (ResultsBuilder('123', 'ecm')
               .as_stage2_workers(4)
               .build())

    assert results['stage2_workers'] == 4


def test_residue_tracking():
    """Test residue file tracking."""
    results = (ResultsBuilder('123', 'ecm')
               .with_residue_file('/path/to/residue.txt')
               .build())

    assert results['residue_file'] == '/path/to/residue.txt'


def test_work_id():
    """Test work_id assignment."""
    results = (ResultsBuilder('123', 'ecm')
               .with_work_id('work-12345')
               .build())

    assert results['work_id'] == 'work-12345'


def test_execution_time():
    """Test execution time tracking."""
    results = (ResultsBuilder('123', 'ecm')
               .with_execution_time(123.45)
               .build())

    assert results['execution_time'] == 123.45


def test_increment_curves():
    """Test incrementing curves_completed."""
    builder = ResultsBuilder('123', 'ecm')
    builder.with_curves(100)
    builder.increment_curves()
    builder.increment_curves(5)

    results = builder.build()
    assert results['curves_completed'] == 6


def test_sigma_parameter():
    """Test sigma parameter."""
    results = (ResultsBuilder('123', 'ecm')
               .with_sigma('123456789')
               .build())

    assert results['sigma'] == '123456789'


def test_results_for_ecm_factory():
    """Test results_for_ecm convenience factory."""
    results = results_for_ecm('123', 50000, 100, param=3).build()

    assert results['composite'] == '123'
    assert results['method'] == 'ecm'
    assert results['b1'] == 50000
    assert results['curves_requested'] == 100
    assert results['parametrization'] == 3
    # Should NOT set b2=0 (that's for stage1 only)
    assert results['b2'] is None


def test_complex_build_scenario():
    """Test a complex real-world scenario."""
    builder = (ResultsBuilder('99999999999999', 'ecm')
               .with_b1(110000000)
               .with_b2(11000000000000)
               .with_curves(3000, 2500)
               .with_parametrization(3)
               .with_work_id('work-abc123')
               .as_multiprocess(16)
               .with_execution_time(1234.56))

    # Simulate adding outputs and finding factors
    builder.add_raw_output('Starting ECM...')
    builder.add_raw_output('Processing curves...')
    builder.with_factors([('12345', '987654321'), ('67890', '111222333')])

    results = builder.build()

    # Verify all fields
    assert results['composite'] == '99999999999999'
    assert results['b1'] == 110000000
    assert results['b2'] == 11000000000000
    assert results['curves_requested'] == 3000
    assert results['curves_completed'] == 2500
    assert results['parametrization'] == 3
    assert results['work_id'] == 'work-abc123'
    assert results['multiprocess'] is True
    assert results['workers'] == 16
    assert results['execution_time'] == 1234.56
    assert results['factor_found'] == '12345'
    assert results['factors_found'] == ['12345', '67890']
    assert results['factor_sigmas'] == {'12345': '987654321', '67890': '111222333'}
    assert 'Starting ECM' in results['raw_output']
    assert 'Processing curves' in results['raw_output']


def test_empty_raw_output():
    """Test build with no raw output."""
    results = ResultsBuilder('123', 'ecm').build()
    assert results['raw_output'] == ''


def test_b2_none_vs_zero():
    """Test distinction between b2=None (default) and b2=0 (stage1 only)."""
    # Default should be None
    results1 = ResultsBuilder('123', 'ecm').build()
    assert results1['b2'] is None

    # Explicit 0 for stage1 only
    results2 = ResultsBuilder('123', 'ecm').with_b2(0).build()
    assert results2['b2'] == 0

    # Explicit value
    results3 = ResultsBuilder('123', 'ecm').with_b2(5000000).build()
    assert results3['b2'] == 5000000
