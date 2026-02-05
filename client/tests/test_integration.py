#!/usr/bin/env python3
"""
Integration tests for key client workflows.

Tests end-to-end workflows with mocked external dependencies:
- Configuration loading and validation
- API client initialization from config
- Work request and submission flow
- Result builder pipeline
- Error handling and recovery

These tests verify component integration without requiring:
- Actual ECM/YAFU binaries
- Running API server
- Network access
"""
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigurationPipeline:
    """Tests for configuration loading and validation pipeline."""

    def test_config_loading_with_defaults(self):
        """Test configuration loads with all expected defaults."""
        from lib.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'client.yaml'
            config_file.write_text("""
client:
  username: testuser
  cpu_name: testcpu

api:
  endpoint: http://localhost:8000
  timeout: 30
  retry_attempts: 3

programs:
  gmp_ecm:
    path: /usr/bin/ecm
    default_curves: 100

execution:
  output_dir: data/outputs

logging:
  level: INFO
  file: data/logs/test.log
""")

            manager = ConfigManager()
            config = manager.load_config(str(config_file))

            assert config['client']['username'] == 'testuser'
            assert config['api']['endpoint'] == 'http://localhost:8000'
            assert config['api']['timeout'] == 30
            assert config['programs']['gmp_ecm']['default_curves'] == 100

    def test_config_local_override_merging(self):
        """Test that client.local.yaml overrides base config."""
        from lib.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Base config
            config_file = Path(tmpdir) / 'client.yaml'
            config_file.write_text("""
client:
  username: baseuser
  cpu_name: basecpu

api:
  endpoint: http://base:8000
  timeout: 30

programs:
  gmp_ecm:
    path: /base/ecm
""")

            # Local override
            local_file = Path(tmpdir) / 'client.local.yaml'
            local_file.write_text("""
client:
  username: localuser

api:
  endpoint: http://local:9000
""")

            manager = ConfigManager()
            config = manager.load_config(str(config_file))

            # Local overrides should take precedence
            assert config['client']['username'] == 'localuser'
            assert config['api']['endpoint'] == 'http://local:9000'
            # Non-overridden values should remain
            assert config['client']['cpu_name'] == 'basecpu'
            assert config['api']['timeout'] == 30


class TestAPIClientIntegration:
    """Tests for API client integration with wrapper."""

    def test_api_client_lazy_initialization(self):
        """Test that API clients are lazily initialized."""
        from lib.base_wrapper import BaseWrapper

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'client.yaml'
            config_file.write_text("""
client:
  username: testuser
  cpu_name: testcpu

api:
  endpoint: http://localhost:8000
  timeout: 30
  retry_attempts: 3

programs:
  gmp_ecm:
    path: /usr/bin/ecm

execution:
  output_dir: data/outputs

logging:
  level: INFO
  file: data/logs/test.log
""")

            wrapper = BaseWrapper(str(config_file))

            # Before initialization, should be None
            assert wrapper.api_client is None

            # Initialize
            wrapper._ensure_api_clients()

            # After initialization, should be set
            assert wrapper.api_client is not None
            assert wrapper.api_client.api_endpoint == 'http://localhost:8000'

    def test_api_client_configuration_propagation(self):
        """Test that API settings propagate correctly to client."""
        from lib.base_wrapper import BaseWrapper

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / 'client.yaml'
            config_file.write_text("""
client:
  username: testuser
  cpu_name: testcpu

api:
  endpoint: http://custom:9999
  timeout: 60
  retry_attempts: 5

programs:
  gmp_ecm:
    path: /usr/bin/ecm

execution:
  output_dir: data/outputs

logging:
  level: INFO
  file: data/logs/test.log
""")

            wrapper = BaseWrapper(str(config_file))
            wrapper._ensure_api_clients()

            assert wrapper.api_client is not None
            assert wrapper.api_client.timeout == 60
            assert wrapper.api_client.retry_attempts == 5


class TestResultsBuilderPipeline:
    """Tests for results builder integration."""

    def test_results_builder_full_pipeline(self):
        """Test building complete results dict through builder."""
        from lib.results_builder import ResultsBuilder

        builder = (ResultsBuilder('123456789', 'ecm')
                   .with_b1(50000)
                   .with_b2(5000000)
                   .with_curves(100, 100)
                   .with_parametrization(3)
                   .with_execution_time(45.2)
                   .with_factors([('12345', '3:111'), ('67890', '3:222')]))

        results = builder.build()

        assert results['composite'] == '123456789'
        assert results['method'] == 'ecm'
        assert results['b1'] == 50000
        assert results['b2'] == 5000000
        assert results['curves_requested'] == 100
        assert results['curves_completed'] == 100
        assert results['parametrization'] == 3
        assert results['execution_time'] == 45.2
        assert '12345' in results['factors_found']
        assert '67890' in results['factors_found']

    def test_results_for_stage1_helper(self):
        """Test results_for_stage1 factory function."""
        from lib.results_builder import results_for_stage1

        builder = results_for_stage1('999888777', b1=110000000, curves=3000, param=3)
        results = builder.build()

        assert results['composite'] == '999888777'
        assert results['b1'] == 110000000
        assert results['b2'] == 0  # Stage 1 only
        assert results['curves_requested'] == 3000
        assert results['parametrization'] == 3


class TestSubmissionPayloadPipeline:
    """Tests for complete submission payload construction."""

    def test_submission_payload_from_results(self):
        """Test building API payload from results dict."""
        from lib.api_client import APIClient

        client = APIClient('http://localhost:8000')

        results = {
            'composite': '123456789',
            'method': 'ecm',
            'b1': 50000,
            'b2': 5000000,
            'curves_requested': 100,
            'curves_completed': 95,
            'execution_time': 30.5,
            'factors_found': ['3', '41152263'],
            'factor_sigmas': {'3': '3:111', '41152263': '3:222'},
            'parametrization': 3,
            'raw_output': 'ECM output here...'
        }

        payload = client.build_submission_payload(
            composite='123456789',
            client_id='user-cpu',
            method='ecm',
            program='gmp-ecm',
            program_version='7.0.4',
            results=results,
            project='test-project'
        )

        # Verify structure
        assert payload['composite'] == '123456789'
        assert payload['client_id'] == 'user-cpu'
        assert payload['project'] == 'test-project'
        assert payload['parameters']['b1'] == 50000
        assert payload['parameters']['b2'] == 5000000
        assert payload['results']['curves_completed'] == 95
        assert payload['results']['execution_time'] == 30.5

    def test_submission_with_multiple_factors(self):
        """Test payload building preserves all factors with sigmas."""
        from lib.api_client import APIClient

        client = APIClient('http://localhost:8000')

        results = {
            'b1': 50000,
            'curves_completed': 50,
            'factors_found': ['7', '11', '13'],
            'factor_sigmas': {'7': '3:100', '11': '3:200', '13': '3:300'}
        }

        payload = client.build_submission_payload(
            composite='1001',
            client_id='test',
            method='ecm',
            program='gmp-ecm',
            program_version='7.0.4',
            results=results
        )

        # Should have factors_found list with sigmas
        factors_list = payload['results']['factors_found']
        assert len(factors_list) == 3

        # Verify each factor has its sigma
        factors_dict = {f['factor']: f['sigma'] for f in factors_list}
        assert factors_dict['7'] == '3:100'
        assert factors_dict['11'] == '3:200'
        assert factors_dict['13'] == '3:300'


class TestFailedSubmissionRecovery:
    """Tests for failed submission persistence and recovery."""

    def test_failed_submission_saves_to_disk(self):
        """Test that failed submissions are persisted."""
        from lib.api_client import APIClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = APIClient('http://localhost:8000')

            results = {
                'composite': '123456789',
                'method': 'ecm',
                'b1': 50000,
                'factor_found': None
            }

            payload = {'composite': '123456789'}

            saved_path = client.save_failed_submission(
                results=results,
                payload=payload,
                output_dir=tmpdir
            )

            assert saved_path is not None
            assert Path(saved_path).exists()

            # Verify contents
            with open(saved_path, 'r') as f:
                saved_data = json.load(f)

            assert saved_data['composite'] == '123456789'
            assert saved_data['submitted'] is False
            assert 'api_payload' in saved_data
            assert 'failed_at' in saved_data

    def test_failed_submission_includes_retry_info(self):
        """Test that failed submissions include retry count."""
        from lib.api_client import APIClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = APIClient('http://localhost:8000', retry_attempts=5)

            results = {'composite': '999'}
            payload = {'composite': '999'}

            saved_path = client.save_failed_submission(
                results=results,
                payload=payload,
                output_dir=tmpdir
            )

            assert saved_path is not None
            with open(saved_path, 'r') as f:
                saved_data = json.load(f)

            assert saved_data['retry_count'] == 5


class TestWorkModeIntegration:
    """Tests for work mode integration with wrapper."""

    def test_work_mode_factory_integration(self):
        """Test work mode factory creates correct mode from args."""
        from lib.work_modes import WorkLoopContext, get_work_mode, StandardAutoWorkMode
        import argparse

        # Create minimal mock wrapper
        wrapper = Mock()
        wrapper.config = {'programs': {'gmp_ecm': {'default_curves': 100}}}
        wrapper.logger = Mock()
        wrapper.interrupted = False
        wrapper.graceful_shutdown_requested = False
        wrapper._ensure_api_clients = Mock()
        wrapper._get_api_client = Mock(return_value=Mock())

        args = argparse.Namespace(
            composite=None,
            stage1_only=False,
            stage2_only=False,
            b1=None,
            b2=None,
            verbose=False,
            multiprocess=False,
            method='ecm',
            project=None
        )

        ctx = WorkLoopContext(
            wrapper=wrapper,
            client_id='test-client',
            args=args,
            work_count_limit=5
        )

        mode = get_work_mode(ctx)

        assert isinstance(mode, StandardAutoWorkMode)
        assert mode.ctx.work_count_limit == 5


class TestECMConfigIntegration:
    """Tests for ECM config object integration."""

    def test_ecm_config_validation(self):
        """Test ECMConfig validates parameters."""
        from lib.ecm_config import ECMConfig

        # Valid config should not raise
        config = ECMConfig(
            composite='123456789',
            b1=50000,
            curves=100
        )

        assert config.composite == '123456789'
        assert config.b1 == 50000
        assert config.curves == 100

    def test_multiprocess_config_auto_workers(self):
        """Test MultiprocessConfig auto-sets workers to CPU count."""
        from lib.ecm_config import MultiprocessConfig
        import multiprocessing as mp

        config = MultiprocessConfig(
            composite='123',
            b1=50000,
            total_curves=100,
            num_processes=None  # Should auto-detect
        )

        assert config.num_processes == mp.cpu_count()

    def test_tlevel_config_defaults(self):
        """Test TLevelConfig has sensible defaults."""
        from lib.ecm_config import TLevelConfig

        config = TLevelConfig(
            composite='123456789',
            target_t_level=35.0
        )

        assert config.start_t_level == 0.0
        assert config.threads == 1
        assert config.use_two_stage is False
        assert config.b2_multiplier == 100.0


class TestFactorResultIntegration:
    """Tests for FactorResult object integration."""

    def test_factor_result_to_dict(self):
        """Test FactorResult converts to dict for API submission."""
        from lib.ecm_config import FactorResult

        result = FactorResult()
        result.add_factor('3', '3:111')
        result.add_factor('7', '3:222')
        result.curves_run = 50
        result.execution_time = 10.5
        result.success = True

        result_dict = result.to_dict('21', 'ecm')

        assert result_dict['composite'] == '21'
        assert result_dict['method'] == 'ecm'
        assert result_dict['curves_completed'] == 50
        assert result_dict['execution_time'] == 10.5
        assert '3' in result_dict['factors_found']
        assert '7' in result_dict['factors_found']

    def test_factor_sigma_pairs(self):
        """Test factor_sigma_pairs property."""
        from lib.ecm_config import FactorResult

        result = FactorResult()
        result.add_factor('3', '3:111')
        result.add_factor('7', None)

        pairs = list(result.factor_sigma_pairs)

        assert len(pairs) == 2
        assert ('3', '3:111') in pairs
        assert ('7', None) in pairs


def main():
    """Run all tests."""
    print("Running integration tests...\n")
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    main()
