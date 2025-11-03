#!/usr/bin/env python3
"""
Tests for subprocess_utils module
"""
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.subprocess_utils import execute_subprocess, execute_subprocess_simple


def test_execute_subprocess_simple():
    """Test simple subprocess execution"""
    stdout, returncode = execute_subprocess_simple(
        cmd=['echo', 'hello world'],
        timeout=5
    )

    assert stdout is not None, "Should return output"
    assert 'hello world' in stdout, "Should contain expected output"
    assert returncode == 0, "Should succeed"

    print("✓ test_execute_subprocess_simple passed")


def test_execute_subprocess_basic():
    """Test basic subprocess execution"""
    result = execute_subprocess(
        cmd=['echo', 'test output'],
        verbose=False
    )

    assert result['stdout'] is not None, "Should return stdout"
    assert 'test output' in result['stdout'], "Should contain expected output"
    assert result['returncode'] == 0, "Should succeed"
    assert not result['terminated_early'], "Should not terminate early"

    print("✓ test_execute_subprocess_basic passed")


def test_execute_subprocess_with_line_callback():
    """Test subprocess execution with line callback"""
    lines_captured = []

    def capture_line(line: str, output_lines: List[str]) -> None:
        lines_captured.append(line)

    result = execute_subprocess(
        cmd=['echo', '-e', 'line1\\nline2\\nline3'],
        verbose=False,
        line_callback=capture_line
    )

    assert len(lines_captured) >= 3, "Should capture multiple lines"
    assert any('line1' in line for line in lines_captured), "Should capture line1"
    assert any('line2' in line for line in lines_captured), "Should capture line2"
    assert any('line3' in line for line in lines_captured), "Should capture line3"

    print("✓ test_execute_subprocess_with_line_callback passed")


def test_execute_subprocess_with_composite():
    """Test subprocess execution with composite input"""
    result = execute_subprocess(
        cmd=['cat'],  # cat will read from stdin
        composite='123456789',
        verbose=False
    )

    assert '123456789' in result['stdout'], "Should receive composite via stdin"

    print("✓ test_execute_subprocess_with_composite passed")


def test_execute_subprocess_timeout():
    """Test subprocess timeout handling"""
    import pytest

    # Test with a command that should timeout
    result = execute_subprocess(
        cmd=['sleep', '10'],
        verbose=False,
        timeout=1  # 1 second timeout for a 10 second sleep
    )

    # Should either timeout or terminate
    # The function handles TimeoutExpired internally
    assert result is not None, "Should return result even on timeout"

    print("✓ test_execute_subprocess_timeout passed")


def test_execute_subprocess_returncode():
    """Test subprocess return code handling"""
    # Test successful command
    result = execute_subprocess(
        cmd=['true'],  # Always returns 0
        verbose=False
    )

    assert result['returncode'] == 0, "Should return 0 for successful command"

    # Test failed command
    result = execute_subprocess(
        cmd=['false'],  # Always returns non-zero
        verbose=False
    )

    assert result['returncode'] != 0, "Should return non-zero for failed command"

    print("✓ test_execute_subprocess_returncode passed")


def test_execute_subprocess_progress_interval():
    """Test progress interval output"""
    # Create a command that produces multiple lines
    # Use printf to avoid issues with echo -e on different systems
    result = execute_subprocess(
        cmd=['sh', '-c', 'for i in 1 2 3 4 5; do echo "Step $i took 100ms"; done'],
        verbose=False,
        progress_interval=2  # Show progress every 2 "Step X took" lines
    )

    assert result is not None, "Should complete successfully"
    assert 'Step' in result['stdout'], "Should contain step output"

    print("✓ test_execute_subprocess_progress_interval passed")


def test_execute_subprocess_log_prefix():
    """Test log prefix functionality"""
    # This test mainly ensures the log_prefix parameter doesn't cause errors
    result = execute_subprocess(
        cmd=['echo', 'test'],
        verbose=False,
        log_prefix="TestPrefix"
    )

    assert result is not None, "Should execute successfully with log prefix"

    print("✓ test_execute_subprocess_log_prefix passed")


def test_execute_subprocess_empty_output():
    """Test handling of commands with no output"""
    result = execute_subprocess(
        cmd=['true'],  # Produces no output
        verbose=False
    )

    assert result is not None, "Should handle empty output"
    assert result['stdout'] == '', "stdout should be empty string"
    assert result['returncode'] == 0, "Should succeed"

    print("✓ test_execute_subprocess_empty_output passed")


def test_execute_subprocess_simple_with_error():
    """Test simple execution with failing command"""
    stdout, returncode = execute_subprocess_simple(
        cmd=['sh', '-c', 'echo error >&2; exit 1'],  # Write to stderr and fail
        timeout=5
    )

    # execute_subprocess_simple returns stdout even on error
    assert stdout is not None, "Should return stdout even on error"
    assert returncode != 0, "Should have non-zero return code"

    print("✓ test_execute_subprocess_simple_with_error passed")


def main():
    """Run all tests"""
    print("Running subprocess_utils tests...\n")

    tests = [
        test_execute_subprocess_simple,
        test_execute_subprocess_basic,
        test_execute_subprocess_with_line_callback,
        test_execute_subprocess_with_composite,
        test_execute_subprocess_timeout,
        test_execute_subprocess_returncode,
        test_execute_subprocess_progress_interval,
        test_execute_subprocess_log_prefix,
        test_execute_subprocess_empty_output,
        test_execute_subprocess_simple_with_error
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
