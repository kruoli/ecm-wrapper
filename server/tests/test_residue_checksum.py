"""
Tests for GMP-ECM residue file checksum verification.

Validates the checksum algorithm from GMP-ECM's ecm/ecm-ecm.h against
real residue files produced by GMP-ECM.

The checksum formula:
    CHKCONST = 4294967291  (from ecm-ecm.h:170)
    checksum = B1 * (sigma % CHKCONST) * (N % CHKCONST) * (X % CHKCONST) * (PARAM + 1) % CHKCONST
"""
import sys
from pathlib import Path

import pytest

# Add server directory to Python path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

# Path to real residue files (relative to repo root)
RESIDUE_DIR = Path(__file__).parent.parent.parent / "client" / "data" / "residues"

CHKCONST = 4294967291  # from GMP-ECM's ecm/ecm-ecm.h:170


def parse_residue_line(line: str) -> dict:
    """Parse a single GMP-ECM residue line into its key-value components."""
    elements = [el.strip() for el in line.split(';') if el.strip()]
    kv_pairs = {}
    for el in elements:
        parts = el.split('=', 1)
        if len(parts) == 2:
            kv_pairs[parts[0]] = parts[1]
    return kv_pairs


def compute_checksum(b1: int, sigma: int, composite: int, x: int, param: int) -> int:
    """
    Compute the GMP-ECM residue checksum.

    This replicates the algorithm from GMP-ECM's ecm/ecm-ecm.h.

    Args:
        b1: Stage 1 bound
        sigma: Curve sigma parameter
        composite: The number being factored (N)
        x: The X coordinate from the residue (stage 1 output)
        param: Parametrization type (0-3)

    Returns:
        Checksum value (mod CHKCONST)
    """
    checksum = b1
    checksum *= sigma % CHKCONST
    checksum *= composite % CHKCONST
    checksum *= x % CHKCONST
    checksum *= param + 1
    checksum %= CHKCONST
    return checksum


def verify_residue_line(line: str, expected_param: int = None) -> bool:
    """
    Verify a single GMP-ECM residue line's checksum.

    Args:
        line: A complete residue line (semicolon-delimited key=value pairs)
        expected_param: If set, verify PARAM matches this value

    Returns:
        True if checksum is valid

    Raises:
        ValueError: If required fields are missing
    """
    kv = parse_residue_line(line)

    required_keys = ['PARAM', 'SIGMA', 'B1', 'N', 'X', 'CHECKSUM']
    missing = [k for k in required_keys if k not in kv]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    param = int(kv['PARAM'])
    sigma = int(kv['SIGMA'])
    b1 = int(kv['B1'])
    n = int(kv['N'])
    x = int(kv['X'], 0)  # X is hex with 0x prefix
    expected_checksum = int(kv['CHECKSUM'])

    if expected_param is not None:
        assert param == expected_param, f"PARAM mismatch: {param} != {expected_param}"

    calculated = compute_checksum(b1, sigma, n, x, param)
    return calculated == expected_checksum


def verify_residue_file(filepath: Path) -> tuple:
    """
    Verify all lines in a GMP-ECM residue file.

    Args:
        filepath: Path to the residue file

    Returns:
        Tuple of (total_lines, passed_lines, failed_lines)
    """
    content = filepath.read_text()
    lines = [
        line.strip() for line in content.strip().split('\n')
        if line.strip() and not line.lstrip().startswith(('[', '#'))
    ]

    total = 0
    passed = 0
    failed = 0

    for line in lines:
        kv = parse_residue_line(line)
        if 'CHECKSUM' not in kv or 'SIGMA' not in kv:
            continue  # Skip non-curve lines

        total += 1
        if verify_residue_line(line):
            passed += 1
        else:
            failed += 1

    return total, passed, failed


# =============================================================================
# Unit tests for the checksum computation
# =============================================================================

class TestChecksumComputation:
    """Test the raw checksum computation with known values."""

    def test_known_values_param3(self):
        """Verify checksum against a known GMP-ECM PARAM=3 residue line."""
        # From 297134095231_17_110M.txt line 1
        b1 = 110000000
        sigma = 804980190
        n = 2108403225778176658472528366989745676172515447880315511907949259471235712524742465283621318043444022069429808626241435248319295958146716534856112507222665012480981881569567696931047
        x = 0x1bdb022a9de2835362562eebe7f57ee5c2d1e23707eedae18918be9fdd650c78792e11fd5665a8a7abc74d9fda5f7c78953f0263b1a926a252685ac4d164455e42a1d7f82cca1fea6cae91
        param = 3
        expected = 1552095609

        assert compute_checksum(b1, sigma, n, x, param) == expected

    def test_known_values_param3_line2(self):
        """Verify checksum against a second known line from same file."""
        # From 297134095231_17_110M.txt line 2
        b1 = 110000000
        sigma = 804980191
        n = 2108403225778176658472528366989745676172515447880315511907949259471235712524742465283621318043444022069429808626241435248319295958146716534856112507222665012480981881569567696931047
        x = 0x2684423163cb8b488861a5a6472c07f83dac6353d5cc4af6950cb9e880e87ef2c648fc9a749cbc97b516e9c99e6672c05310cc35afa7c557164f00bec56d6595dd0c36046003e8e6525575
        param = 3
        expected = 3391081471

        assert compute_checksum(b1, sigma, n, x, param) == expected

    def test_known_values_different_composite(self):
        """Verify checksum with a different composite number."""
        # From 13462149171001_17_1.txt line 1
        b1 = 110000000
        sigma = 1660499024
        n = 1898322094601691287671083579644648960622409614436961148813460752956661343305009370822319193803729141102686477466619494745317621158684857175197808975295265123106944330215873257354044399847905145122173501088509
        x = 0x4c5cbaf8629e69edf098f321192986ce6ed48dfe0ad377d6061305c99c4df05f2dba3618104577b035a7214152bd24b987aa09e82699855f27609524c14c926e75b4fcbcc78214dc3defcf18522f788990d9eb96134c
        param = 3
        expected = 4119582023

        assert compute_checksum(b1, sigma, n, x, param) == expected

    def test_known_values_higher_b1(self):
        """Verify checksum with B1=330000000."""
        # From 203282179468878772294821547733021360352612730150400763049_5.txt line 1
        b1 = 330000000
        sigma = 2087464396
        n = 31075270644091267088664616812064881861728699030126775171792305701936499464476903069443979826236333992428677250317653509057559472655353411217775132694777701339580840409598226971
        x = 0x6b9ad3e99ef6f37fd1142225418d45f669cc02c2982211af2b70fd27563308e258f2a38294fda29ca276364873cb35ee203dc88698cca3349a341baf041199b2a7e5f914f22b8d4445
        param = 3
        expected = 84299641

        assert compute_checksum(b1, sigma, n, x, param) == expected

    def test_chkconst_value(self):
        """Verify CHKCONST is the expected prime near 2^32."""
        assert CHKCONST == 4294967291
        # It's the largest prime below 2^32
        assert CHKCONST < 2**32
        assert CHKCONST > 2**32 - 10


class TestResidueLineParsing:
    """Test parsing of individual residue lines."""

    def test_parse_standard_line(self):
        """Parse a standard GMP-ECM residue line."""
        line = (
            "METHOD=ECM; PARAM=3; SIGMA=804980190; B1=110000000; "
            "N=123456789; X=0xdeadbeef; CHECKSUM=42; "
            "PROGRAM=GMP-ECM 7.0.6; X0=0x0; Y0=0x0"
        )
        kv = parse_residue_line(line)
        assert kv['METHOD'] == 'ECM'
        assert kv['PARAM'] == '3'
        assert kv['SIGMA'] == '804980190'
        assert kv['B1'] == '110000000'
        assert kv['N'] == '123456789'
        assert kv['X'] == '0xdeadbeef'
        assert kv['CHECKSUM'] == '42'

    def test_parse_handles_whitespace(self):
        """Parsing handles extra whitespace in values."""
        line = "PARAM=3 ; SIGMA= 100 ; B1=50000"
        kv = parse_residue_line(line)
        assert kv['PARAM'] == '3'
        # Note: leading space preserved in value after split('=', 1)
        assert kv['SIGMA'].strip() == '100'

    def test_missing_field_raises(self):
        """verify_residue_line raises ValueError for missing fields."""
        line = "METHOD=ECM; PARAM=3; SIGMA=100"
        with pytest.raises(ValueError, match="Missing required fields"):
            verify_residue_line(line)


# =============================================================================
# Integration tests against real residue files
# =============================================================================

def get_residue_files():
    """Get all residue files from the client data directory."""
    if not RESIDUE_DIR.exists():
        return []
    return sorted(RESIDUE_DIR.glob("*.txt"))


# Collect residue files for parametrize
_residue_files = get_residue_files()


@pytest.mark.skipif(not _residue_files, reason="No residue files found in client/data/residues/")
class TestRealResidueFiles:
    """Verify checksums against actual GMP-ECM residue files."""

    @pytest.mark.parametrize(
        "residue_file",
        _residue_files,
        ids=[f.name for f in _residue_files]
    )
    def test_all_checksums_valid(self, residue_file):
        """Every line in every residue file should pass checksum verification."""
        total, passed, failed = verify_residue_file(residue_file)

        if total == 0:
            pytest.skip(f"No checksum lines found in {residue_file.name}")

        assert failed == 0, (
            f"{residue_file.name}: {failed}/{total} lines failed checksum verification"
        )
        assert passed == total
        assert total > 0

    def test_at_least_some_files_found(self):
        """Sanity check: we should have residue files to test."""
        assert len(_residue_files) > 0, (
            f"Expected residue files in {RESIDUE_DIR}"
        )

    def test_sample_file_line_count(self):
        """Verify a known file has expected curve count."""
        sample = RESIDUE_DIR / "297134095231_17_110M.txt"
        if not sample.exists():
            pytest.skip("Sample file not found")

        total, passed, failed = verify_residue_file(sample)
        assert total == 3072  # Known line count
        assert passed == 3072
        assert failed == 0
