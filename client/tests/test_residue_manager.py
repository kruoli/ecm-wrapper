#!/usr/bin/env python3
"""
Quick tests for ResidueFileManager functionality
"""
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.residue_manager import ResidueFileManager

def test_parse_metadata():
    """Test parsing residue file metadata"""
    mgr = ResidueFileManager()

    # Create a sample residue file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.res') as f:
        f.write("N=123456789012345678901234567890 B1=50000 METHOD=ECM SIGMA=12345\n")
        f.write("METHOD=ECM SIGMA=67890\n")
        f.write("METHOD=ECM SIGMA=11111\n")
        test_file = f.name

    try:
        # Test parsing
        result = mgr.parse_metadata(test_file)
        assert result is not None, "parse_metadata should return a result"

        composite, b1, curve_count = result
        assert composite == "123456789012345678901234567890", f"Expected composite, got {composite}"
        assert b1 == 50000, f"Expected b1=50000, got {b1}"
        assert curve_count == 3, f"Expected 3 curves, got {curve_count}"

        print("✓ test_parse_metadata passed")
        pass  # Test passed
    finally:
        os.unlink(test_file)

def test_split_into_chunks():
    """Test splitting residue file into chunks"""
    mgr = ResidueFileManager()

    # Create a sample residue file with multiple curves
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.res') as f:
        f.write("N=123456789 B1=50000 METHOD=ECM CHECKSUM=abc\n")
        for i in range(10):
            f.write(f"SIGMA={10000 + i}\n")
            f.write(f"X=12345678{i}\n")
        test_file = f.name

    try:
        # Create temp directory for chunks
        with tempfile.TemporaryDirectory() as chunk_dir:
            # Split into 3 chunks
            chunk_files = mgr.split_into_chunks(test_file, 3, chunk_dir)

            assert len(chunk_files) > 0, "Should create at least one chunk"
            assert len(chunk_files) <= 3, f"Should create at most 3 chunks, got {len(chunk_files)}"

            # Verify chunks exist
            for chunk_file in chunk_files:
                assert os.path.exists(chunk_file), f"Chunk file should exist: {chunk_file}"

                # Verify chunk has header
                with open(chunk_file, 'r', encoding='utf-8') as cf:
                    content = cf.read()
                    assert 'N=123456789' in content, "Chunk should contain header with N"
                    assert 'B1=50000' in content, "Chunk should contain header with B1"

            print(f"✓ test_split_into_chunks passed (created {len(chunk_files)} chunks)")
    finally:
        os.unlink(test_file)

def test_parse_metadata_p95_hex_format():
    """Test parsing P95/mprime residue files with hex N and no B1"""
    mgr = ResidueFileManager()

    # Sample P95/mprime format: PARAM=0; N=0x<hex>; QX=0x<hex>; SIGMA=<decimal>
    # N=0xFF = 255
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.res') as f:
        f.write("PARAM=0; N=0xFF; QX=0xAB; SIGMA=12345\n")
        f.write("PARAM=0; N=0xFF; QX=0xCD; SIGMA=67890\n")
        f.write("PARAM=0; N=0xFF; QX=0xEF; SIGMA=11111\n")
        test_file = f.name

    try:
        result = mgr.parse_metadata(test_file)
        assert result is not None, "parse_metadata should handle P95/mprime hex format"

        composite, b1, curve_count = result
        assert composite == "255", f"Expected composite=255 (0xFF as decimal), got {composite}"
        assert b1 == 0, f"Expected b1=0 (not present in P95 format), got {b1}"
        assert curve_count == 3, f"Expected 3 curves, got {curve_count}"

        print("✓ test_parse_metadata_p95_hex_format passed")
    finally:
        os.unlink(test_file)

def test_split_p95_hex_format():
    """Test splitting P95/mprime residue files into chunks"""
    mgr = ResidueFileManager()

    # Create P95/mprime format residue file with 6 curves
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.res') as f:
        for i in range(6):
            f.write(f"PARAM=0; N=0xFF; QX=0x{i:02X}; SIGMA={10000 + i}\n")
        test_file = f.name

    try:
        with tempfile.TemporaryDirectory() as chunk_dir:
            chunk_files = mgr.split_into_chunks(test_file, 3, chunk_dir)

            assert len(chunk_files) == 3, f"Expected 3 chunks, got {len(chunk_files)}"

            # Count total curves across all chunks
            total_curves = 0
            for chunk_file in chunk_files:
                assert os.path.exists(chunk_file), f"Chunk file should exist: {chunk_file}"
                with open(chunk_file, 'r') as cf:
                    lines = [l for l in cf.readlines() if l.strip()]
                    total_curves += len(lines)
                    # Each line should be a complete curve
                    for line in lines:
                        assert 'PARAM=0' in line, "Each line should be a complete P95 curve"
                        assert 'SIGMA=' in line, "Each line should contain SIGMA"

            assert total_curves == 6, f"Expected 6 total curves across chunks, got {total_curves}"

            print(f"✓ test_split_p95_hex_format passed (created {len(chunk_files)} chunks)")
    finally:
        os.unlink(test_file)

def test_correlate_factor_to_sigma_hex():
    """Test correlating factor to sigma with hex N format"""
    mgr = ResidueFileManager()

    # 0xDD = 221 = 13 * 17
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.res') as f:
        f.write("PARAM=0; N=0xDD; QX=0xAB; SIGMA=12345\n")
        f.write("PARAM=0; N=0xDD; QX=0xCD; SIGMA=67890\n")
        test_file = f.name

    try:
        sigma = mgr.correlate_factor_to_sigma("13", test_file)
        assert sigma is not None, "Should find sigma for valid factor of 0xDD (221)"
        assert sigma == "12345", f"Expected sigma=12345, got {sigma}"

        sigma = mgr.correlate_factor_to_sigma("23", test_file)
        assert sigma is None, "Should not find sigma for non-factor"

        print("✓ test_correlate_factor_to_sigma_hex passed")
    finally:
        os.unlink(test_file)

def test_correlate_factor_to_sigma():
    """Test correlating factor to sigma value"""
    mgr = ResidueFileManager()

    # Create a sample residue file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.res') as f:
        f.write("N=221 B1=50000 METHOD=ECM SIGMA=12345\n")  # 221 = 13 * 17
        f.write("METHOD=ECM SIGMA=67890\n")
        test_file = f.name

    try:
        # Test correlation with a valid factor
        sigma = mgr.correlate_factor_to_sigma("13", test_file)
        assert sigma is not None, "Should find a sigma value"
        assert sigma == "12345", f"Expected sigma=12345, got {sigma}"

        # Test with invalid factor
        sigma = mgr.correlate_factor_to_sigma("23", test_file)
        assert sigma is None, "Should not find sigma for non-factor"

        print("✓ test_correlate_factor_to_sigma passed")
        pass  # Test passed
    finally:
        os.unlink(test_file)

def main():
    """Run all tests"""
    print("Running ResidueFileManager tests...\n")

    tests = [
        test_parse_metadata,
        test_parse_metadata_p95_hex_format,
        test_split_into_chunks,
        test_split_p95_hex_format,
        test_correlate_factor_to_sigma,
        test_correlate_factor_to_sigma_hex,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0

if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
