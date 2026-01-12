# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECM Coordination Middleware - a minimal, focused system for coordinating distributed ECM factorization work:
- **Client components**: Standalone Python wrappers for GMP-ECM and YAFU factorization
- **Server component**: FastAPI-based coordination middleware with PostgreSQL backend
- **Architecture**: Lightweight middleware that projects can integrate for ECM work coordination
- **Focus**: Pure ECM coordination, t-level progress tracking, and work assignment

## Development Commands

### Client Development
```bash
# Install client dependencies
pip install requests pyyaml

# Run ECM factorization with GMP-ECM
python3 client/ecm_wrapper.py --composite "123456789012345" --curves 100 --b1 50000

# Run YAFU factorization (various modes)
python3 client/yafu_wrapper.py --composite "123456789012345" --mode ecm --curves 100
python3 client/yafu_wrapper.py --composite "123456789012345" --mode pm1 --b1 1000000
python3 client/yafu_wrapper.py --composite "123456789012345" --mode auto

# Test without API submission (ecm_wrapper.py defaults to no submission)
python3 client/ecm_wrapper.py --composite "123456789012345"

# Submit results to API (opt-in for ecm_wrapper.py)
python3 client/ecm_wrapper.py --composite "123456789012345" --submit

# Stage 1 only with residue upload to server (manual mode)
python3 client/ecm_wrapper.py --composite "123456789012345" --b1 50000 --curves 100 --stage1-only --upload

# Auto-work mode - continuously request and process work from server
python3 client/ecm_client.py                    # Use server's target t-levels
python3 client/ecm_client.py --work-count 5     # Process 5 assignments then exit
python3 client/ecm_client.py --tlevel 35        # Override with client t-level
python3 client/ecm_client.py --b1 50000 --b2 5000000 --curves 100  # Override with B1/B2
python3 client/ecm_client.py --b1 26e7 --b2 4e11 --curves 100      # Scientific notation support
python3 client/ecm_client.py --two-stage --b1 50000 --b2 5000000   # GPU two-stage mode
python3 client/ecm_client.py --multiprocess --workers 8            # Multiprocess mode
python3 client/ecm_client.py --min-digits 60 --max-digits 80       # Filter by size

# Decoupled two-stage mode - separate GPU and CPU workers
python3 client/ecm_client.py --stage1-only --b1 110000000 --curves 3000 --gpu  # GPU producer
python3 client/ecm_client.py --stage2-only --b2 11000000000000 --workers 8  # CPU consumer

# Run batch processing scripts
cd client/scripts/
./run_batch.sh                    # ECM batch
./run_pm1_batch.sh               # GMP-ECM P-1 batch
./run_pm1_batch_yafu.sh          # YAFU P-1 batch

# Resend failed submissions
python3 resend_failed.py --dry-run  # Test without marking files
python3 resend_failed.py            # Submit and mark as completed
```

### Server Development

**Quick Start (Recommended)**
```bash
cd server/

# Start PostgreSQL (uses existing data volume)
docker-compose -f docker-compose.dev.yml up -d postgres

# Start API server
source venv/bin/activate
export DATABASE_URL="postgresql://ecm_user:ecm_password@localhost:5434/ecm_distributed"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Setup Commands**
```bash
# Install server dependencies (first time only)
cd server/
pip install -r requirements.txt

# Database operations
alembic revision --autogenerate -m "Description"  # Create new migration
alembic upgrade head                               # Apply migrations
alembic downgrade -1                              # Rollback one migration
```

**Alternative Setup (Local PostgreSQL)**
```bash
# Set up local database on default port 5432
createdb ecm_distributed
createuser ecm_user -P  # password: ecm_password

# Start with local PostgreSQL
export DATABASE_URL="postgresql://ecm_user:ecm_password@localhost:5432/ecm_distributed"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing and Validation
```bash
# Test server health
curl http://localhost:8000/health

# View API documentation
# Open: http://localhost:8000/docs

# Monitor client logs
tail -f client/scripts/ecm_client.log

# Test API endpoints
curl http://localhost:8000/api/v1/composites
curl -X POST http://localhost:8000/api/v1/results/ecm \
  -H "Content-Type: application/json" \
  -d '{"client_id": "test", "composite": "123", "factors": ["3", "41"]}'

# Run unit tests
cd server
source venv/bin/activate
pytest tests/test_number_utils.py -v           # Test number utilities
pytest tests/ -v  # All server tests

cd ../client
pytest test_factorization.py -v               # Test parsing logic
```

### Server Refactoring Documentation

**IMPORTANT:** The server codebase underwent significant refactoring (Phase 1 & 2) on 2025-10-21:

- **[REFACTORING_GUIDE.md](./server/REFACTORING_GUIDE.md)** - Complete migration guide with examples
- **[REFACTORING_QUICK_REFERENCE.md](./server/REFACTORING_QUICK_REFERENCE.md)** - Quick lookup for new patterns

**Key changes:**
- ✅ Unified service architecture with dependency injection
- ✅ Centralized error handling utilities
- ✅ Centralized calculation utilities
- ✅ Eliminated 300-400 lines of duplicate code
- ✅ All routes use dependency injection (no module-level singletons)

**If you're adding new routes or services, follow the patterns in the refactoring guide.**

## New Features (2025-11)

### Auto-Work Mode
Clients can now continuously request and process work assignments from the server without manually specifying composites:

**Implementation:**
- **Client**: `ecm_client.py` provides auto-work mode (implied, no flag needed)
- **API**: Uses `/ecm-work` endpoint to request assignments
- **Work Lifecycle**: Automatic claim → execute → submit → complete workflow

**Features:**
- **Server t-level mode (default)**: Uses server's target_t_level and current_t_level from work assignment
- **Client override modes**: Override with `--b1/--b2` or `--tlevel`
- **Work count limit**: `--work-count N` to process N assignments then exit
- **Filtering**: `--min-digits`, `--max-digits`, `--priority` to filter work
- **Mode support**: Compatible with `--multiprocess` and `--two-stage` (B1/B2 mode only)
- **Graceful shutdown**: Multi-level Ctrl+C handling (see "Graceful Shutdown" section below)

**API Methods** (`client/lib/api_client.py`):
- `get_ecm_work()` - Request work from `/ecm-work` endpoint
- `complete_work()` - Mark work complete via `POST /work/{work_id}/complete`
- `abandon_work()` - Release work via `DELETE /work/{work_id}`
- `upload_residue()` - Upload residue file to server (decoupled two-stage)
- `get_residue_work()` - Request stage 2 work from residue pool
- `download_residue()` - Download residue file for stage 2 processing
- `complete_residue()` - Mark stage 2 complete, supersede stage 1
- `abandon_residue()` - Release residue claim

**Example workflows:**
```bash
# Simple: Use server t-levels, run until stopped
python3 ecm_client.py

# Batch: Process 10 assignments
python3 ecm_client.py --work-count 10

# Custom params with multiprocess
python3 ecm_client.py --tlevel 35 --multiprocess --workers 8
```

### Decoupled Two-Stage ECM (2025-11)
GPU and CPU workers can now run stage 1 and stage 2 **independently** for maximum resource utilization:

**Architecture:**
- **Stage 1 Producer** (GPU): Runs stage 1 only, uploads residue file to server
- **Stage 2 Consumer** (CPU): Downloads residue, runs stage 2, reports results
- **Automatic supersession**: Stage 2 completion supersedes Stage 1 attempt for accurate t-level accounting

**Implementation:**
- **Client flags**: `--stage1-only` and `--stage2-only`
- **Server endpoints** (`server/app/api/v1/residues.py`):
  - `POST /residues/upload` - Upload stage 1 residue file
  - `GET /residues/work` - Request stage 2 work
  - `GET /residues/{id}/download` - Download residue file
  - `POST /residues/{id}/complete` - Mark complete, supersede stage 1
  - `DELETE /residues/{id}/claim` - Release residue claim
- **Database**: New `ecm_residues` table, `ecm_attempts.superseded_by` column
- **T-level accounting**: Excludes superseded attempts when calculating progress

**Workflow:**
```
Stage 1 (GPU):
1. Request regular ECM work → Composite + target params
2. Run stage 1 (B2=0), submit results → Get attempt_id
3. Upload residue file → Links to stage1_attempt_id
4. Server credits t-level for stage 1 work

Stage 2 (CPU):
1. Request residue work → Composite + B1 + residue info
2. Download residue file
3. Run stage 2, submit results → Get stage2_attempt_id
4. Complete residue → Marks stage 1 as superseded
5. Server recalculates t-level (excludes stage 1, counts full stage 2)
```

**Usage examples:**
```bash
# GPU worker: Run stage 1 only, upload residues
python3 ecm_client.py --stage1-only \
  --b1 110000000 --curves 3000 --gpu

# CPU worker: Process stage 2 from residue pool
python3 ecm_client.py --stage2-only \
  --b2 11000000000000 --workers 8

# With work limits and filtering
python3 ecm_client.py --stage1-only \
  --b1 26e7 --curves 1000 --work-count 10 --min-digits 70

python3 ecm_client.py --stage2-only \
  --b2 4e11 --max-digits 90 --priority 5
```

**Benefits:**
- GPU workers maximize throughput on stage 1 (no stage 2 bottleneck)
- CPU workers handle memory-intensive stage 2 efficiently
- Flexible resource allocation (different GPU/CPU ratios)
- Residues stored on server (1.4-1.7 MB per 3000 curves)
- Auto-cleanup after 7 days if not consumed

### Google Colab Support
New `colab_setup.ipynb` notebook for running ECM client in Google Colab:
- One-click setup with username input
- Automatic ECM binary download from GitHub releases
- Pre-configured with production API endpoint
- GPU acceleration enabled by default
- Instructions for batch processing via file upload

## Architecture Overview

### ECM Coordination Model
```
┌─────────────────────┐    HTTP/API     ┌─────────────────────┐
│   Client (Python)  │◄──────────────►│ ECM Middleware      │
│   • GMP-ECM        │                 │   • Work assignment │
│   • YAFU           │                 │   • T-level tracking│
│   • Batch scripts  │                 │   • Progress monitor│
└─────────────────────┘                 └─────────────────────┘
                                                 │
                                        ┌───────────────────┐
                                        │ Any Project Can   │
                                        │ Submit Numbers    │
                                        │ Get Results       │
                                        └───────────────────┘
```

### Key Components

#### Client Components
- **ECMWrapper** (client/ecm_wrapper.py:14): GMP-ECM execution with multiple modes
  - Standard mode, two-stage GPU/CPU, multiprocess, t-level targeting
  - Curve-by-curve control and two-stage processing
  - Multiple factor handling with automatic deduplication
- **YAFUWrapper** (client/yafu_wrapper.py:8): YAFU multi-method factorization coordination
  - ECM, P-1, P+1, SIQS, NFS, and automatic mode selection
  - Thread pool configuration
- **CADOWrapper** (client/cado_wrapper.py): Number Field Sieve for large numbers
- **AliquotWrapper** (client/aliquot_wrapper.py): Aliquot sequence calculator with FactorDB integration
- **BaseWrapper** (client/lib/base_wrapper.py:17): Shared base class
  - Configuration loading via ConfigManager
  - API client initialization (supports multiple endpoints)
  - Subprocess execution with timeout handling
  - Result parsing and factor logging

#### Server Components
- **API Server** (server/app/main.py): FastAPI middleware with coordination endpoints
- **Database Models** (server/app/models/): Minimal schema focused on ECM coordination
- **API Routes** (server/app/api/v1/): RESTful endpoints for work assignment and results
- **T-Level Services** (server/app/services/): T-level calculation and progress tracking

### Configuration System
- **client.yaml**: Default client configuration (API endpoints, binary paths, default parameters)
  - Contains sensible defaults that work out of the box
  - Checked into git for version control
- **client.local.yaml**: Local overrides for client.yaml (gitignored, machine-specific settings)
  - Deep merges with client.yaml (local settings override defaults)
  - Use `client.local.yaml.example` as template
  - Auto-detected by BaseWrapper and arg_parser
  - **Important**: Always pass `client.yaml` as config path - BaseWrapper handles the merge
- **resend_failed.py**: Inherits from BaseWrapper to reuse config loading logic
- **server/app/config.py**: Server configuration (database URL, API settings, residue storage path)
  - `RESIDUE_STORAGE_PATH`: Directory for storing residue files (default: `data/residues`)
- **docker-compose.yml**: Full system deployment configuration
- **alembic.ini**: Database migration configuration

### Middleware Architecture Layers
```
┌─────────────────────────────────────────────────────────────┐
│                   ECM Coordination API                     │
├─────────────────────────────────────────────────────────────┤
│                      Core Endpoints                        │
│  /composites  /submit  /work  /residues  /admin  /stats    │
├─────────────────────────────────────────────────────────────┤
│                   Minimal Services                         │
│  WorkAssignment  TLevelCalculation  ResidueManager  Dash   │
├─────────────────────────────────────────────────────────────┤
│                   Simplified Models                        │
│  Composites  ECMAttempts  Factors  ECMResidues  Clients    │
├─────────────────────────────────────────────────────────────┤
│                    PostgreSQL Database                     │
└─────────────────────────────────────────────────────────────┘
```

## ECM Coordination Workflow

1. **Project submits numbers**: Upload composites with target t-levels via API or admin interface
   - Bulk upload via CSV/text file (`/admin/composites/upload`)
   - Structured upload with metadata (`/admin/composites/bulk-structured`)
   - Can update existing composites: `current_composite`, `priority`, `is_fully_factored`, `is_prime`, `has_snfs_form`, `snfs_difficulty`
2. **Work assignment**: Clients request ECM work assignments with optimal B1/B2 parameters
3. **Client execution**: Wrapper scripts execute GMP-ECM/YAFU binaries with assigned parameters
4. **Progress tracking**: T-level progress updated as curves complete via `/submit_result`
5. **Factor discovery**: Numbers marked as factored when factors found
   - **Group order calculation**: Elliptic curve group orders automatically calculated via PARI/GP
   - Supports all parametrizations (0, 1, 2, 3) with proper curve construction
   - Prime factorization of group order computed for mathematical analysis
6. **Result delivery**: Projects retrieve factorization results via API
7. **Manual curve submission**: Upload ECM curves via `/submit_result` endpoint with full metadata

## Minimal Database Schema

Essential tables for ECM coordination:
- `composites`: Numbers with t-level progress (id, number, digit_length, target_t_level, current_t_level, prior_t_level, is_prime, is_fully_factored, priority)
  - `prior_t_level`: Work done before import (used as starting point for t-level calculations via `-w` flag)
  - `current_t_level`: Actual combined t-level including prior work (calculated properly, not simple addition)
- `ecm_attempts`: Individual ECM curve attempts with B1/B2 parameters and parametrization (0-3)
  - `superseded_by`: Links stage 1 attempt to stage 2 that replaced it (for decoupled two-stage)
- `ecm_residues`: Stage 1 residue files for decoupled two-stage ECM (NEW 2025-11)
  - `composite_id`, `stage1_attempt_id`: Links to composite and original stage 1 attempt
  - `b1`, `parametrization`, `curve_count`: Parsed from residue file
  - `storage_path`, `file_size_bytes`, `checksum`: File storage metadata
  - `status`: 'available', 'claimed', 'completed', 'expired'
  - `claimed_by`, `claimed_at`, `expires_at`: Lifecycle tracking
- `factors`: Discovered factors with discovery methods, sigma values, and elliptic curve group orders
  - `sigma`: Sigma value that found this factor (for reproducibility)
  - `group_order`: Calculated elliptic curve group order (via PARI/GP)
  - `group_order_factorization`: Prime factorization of the group order
- `work_assignments`: Active work assignments to clients
- `clients`: Registered client information and capabilities
- `projects`: Optional organizational structure for campaigns

### Key Schema Updates
- **ecm_attempts.parametrization**: ECM parametrization type (0, 1, 2, or 3) - affects t-level calculations
  - Parametrization 1: Montgomery curves (CPU default)
  - Parametrization 3: Twisted Edwards curves (GPU default)
- **ecm_attempts.superseded_by**: When stage 2 completes, stage 1 attempt is marked as superseded
  - T-level calculator excludes superseded attempts to avoid double-counting
- **composites.prior_t_level**: T-level work done before import into the system
  - Used as starting point via `-w` flag when calculating current t-level
  - T-levels are logarithmic/probabilistic, not additive (t40 + t40 ≈ t41, not t80)
  - `current_t_level` represents the true combined t-level from prior + new work
- **factors.sigma**: Sigma value that found the factor (for reproducibility)
- **ecm_attempts.b2**: Can be NULL (use GMP-ECM default) or 0 (stage 1 only)

## Binary Dependencies

### Client Dependencies
- **GMP-ECM**: Configure path in client.yaml `programs.gmp_ecm.path`
- **YAFU**: Configure path in client.yaml `programs.yafu.path`
- Both programs must be compiled and accessible on client machines

### Server Dependencies
- **t-level binary**: Deployed to `server/bin/t-level` for t-level calculations
- **PARI/GP**: Installed in Docker container for elliptic curve group order calculations
- **group.gp script**: PARI/GP script deployed to `server/bin/group.gp` for FindGroupOrder function

## Client Implementation Details

### V2 API (Config-Based Architecture)

The ECM wrapper uses a modern config-based API (v2) with typed configuration objects and strongly-typed return values:

#### Configuration Classes (`lib/ecm_config.py`)

**ECMConfig** - Standard ECM execution:
```python
from lib.ecm_config import ECMConfig

config = ECMConfig(
    composite="123456789012345",
    b1=50000,
    b2=5000000,         # Optional, None = GMP-ECM default
    curves=100,
    sigma=None,          # Optional: int or "3:N" string format
    parametrization=1,   # 1=CPU (Montgomery), 3=GPU (Twisted Edwards)
    threads=1,
    verbose=False,
    use_gpu=False,       # Auto-sets parametrization=3 if True
    method="ecm"         # 'ecm', 'pm1', 'pp1'
)
# Note: timeout removed - ECM can run for extended periods

result = wrapper.run_ecm_v2(config)
```

**TwoStageConfig** - GPU stage 1 + CPU stage 2:
```python
from lib.ecm_config import TwoStageConfig

config = TwoStageConfig(
    composite="123456789012345",
    b1=110000000,
    b2=11000000000000,
    stage1_curves=3000,
    stage2_curves_per_residue=1000,
    stage1_parametrization=3,  # GPU
    stage2_parametrization=1,  # CPU
    threads=8,                  # Stage 2 workers
    save_residues="residues/output.txt",  # Optional path
    no_submit=False             # For testing
)

result = wrapper.run_two_stage_v2(config)
```

**MultiprocessConfig** - Parallel CPU workers:
```python
from lib.ecm_config import MultiprocessConfig

config = MultiprocessConfig(
    composite="123456789012345",
    b1=50000,
    total_curves=1000,
    curves_per_process=100,
    num_processes=8,       # Auto-detects CPU count if None
    parametrization=1,     # CPU only
    verbose=False
)

result = wrapper.run_multiprocess_v2(config)
```

**TLevelConfig** - Progressive t-level targeting:
```python
from lib.ecm_config import TLevelConfig

config = TLevelConfig(
    composite="123456789012345",
    target_t_level=35.0,
    b1_strategy="optimal",  # 'optimal', 'conservative', 'aggressive'
    parametrization=1,       # Auto-switches to 3 if use_two_stage=True
    threads=1,               # workers=8 for multiprocess
    use_two_stage=False,     # Enable GPU two-stage mode
    project="my-project",    # For API submission
    no_submit=False,
    work_id=None             # For auto-work batch submissions
)

result = wrapper.run_tlevel_v2(config)
```

#### FactorResult Return Type (`lib/ecm_config.py:166`)

All v2 methods return a `FactorResult` object instead of a dictionary:

```python
from lib.ecm_config import FactorResult

result = wrapper.run_ecm_v2(config)

# Access results
print(f"Success: {result.success}")           # bool
print(f"Factors: {result.factors}")           # List[str]
print(f"Sigmas: {result.sigmas}")             # List[Optional[str]]
print(f"Curves: {result.curves_run}")         # int
print(f"Time: {result.execution_time:.2f}s")  # float
print(f"Output: {result.raw_output}")         # Optional[str]

# Get factor/sigma pairs
for factor, sigma in result.factor_sigma_pairs:
    print(f"Factor {factor} found with sigma {sigma}")

# Add factors programmatically
result.add_factor("123", "3:12345")
```

#### Key Benefits of V2 API

- **Type safety**: Config validation at construction time
- **Cleaner interfaces**: 4 parameters vs 10-15 function arguments
- **Single source of truth**: All methods use `_execute_ecm_primitive()`
- **Strongly typed returns**: `FactorResult` vs untyped dicts
- **Easier testing**: Config objects are easy to construct and validate

#### Config Validation (ECMConfigValidation Mixin)

All config classes inherit from `ECMConfigValidation` mixin which provides shared validation:
- `_validate_composite(composite)` - Ensures composite is non-empty
- `_validate_b1(b1)` - Ensures B1 is positive
- `_validate_method(method)` - Validates method is 'ecm', 'pm1', or 'pp1'
- `_validate_parametrization(param)` - Validates param is 0-3

#### WorkMode Pattern (lib/work_modes.py)

Auto-work mode (`ecm_client.py`) uses a Strategy pattern for different work modes:
- **StandardAutoWorkMode** - Regular ECM work from server assignments
- **Stage1ProducerMode** - GPU stage 1 only, uploads residues to server
- **Stage2ConsumerMode** - Downloads residues from server, runs stage 2

Each mode implements the `WorkMode` abstract base class with template method `run()`:
```python
class WorkMode(ABC):
    def run(self) -> int:
        while self.should_continue():
            work = self.request_work()
            result = self.execute_work(work)
            self.submit_results(work, result)
            self.complete_work(work)
        return self.completed_count
```

#### T-Level Calculation Details

T-level mode uses a hybrid approach for optimal performance:

1. **Cached transitions** (`lib/ecm_math.py:168`): Standard 5-digit increments (t20→t25, t25→t30, etc.)
   - Separate caches for parametrization 1 (CPU) and 3 (GPU)
   - Example: `(20, 25, 1): (50000, 261)` means 261 curves at B1=50000 for CPU

2. **Direct t-level binary calls**: Non-standard targets (e.g., t35→t38.75)
   - Uses `calculate_curves_to_target_direct()` with `-w`, `-t`, `-b`, `-p` flags
   - Exact calculations from the t-level binary itself

3. **Parametrization awareness**: p=1 (CPU) and p=3 (GPU) give different t-levels for same curves
   - 107 curves @ B1=11000, p=1 → t20.012
   - 109 curves @ B1=11000, p=3 → t20.012 (requires more curves)

### Output Parsing
- **ECM**: Multiple factor detection with prime factor filtering using `parse_ecm_output_multiple()` (lib/parsing_utils.py:61)
  - Pattern: `r'Factor found in step \d+: (\d+)'`
  - Composite factor filtering (avoids submitting products of known primes)
  - Multi-pattern matching with deduplication
- **YAFU**: Unified parsing functions `parse_yafu_ecm_output()` and `parse_yafu_auto_factors()` (lib/parsing_utils.py:141-200)
  - Multiple patterns for P/Q notation and factor formats
  - Supports ECM, SIQS, NFS output formats
- **Shared Infrastructure**: Unified subprocess execution via `BaseWrapper.run_subprocess_with_parsing()` (lib/base_wrapper.py:278)
  - Single execution path for all wrappers
  - Consistent error handling and timeout management

### Results Dictionary Construction
- **ResultsBuilder** (`client/lib/results_builder.py`): Unified builder for constructing ECM results dictionaries
  - **Purpose**: Eliminates code duplication across 7+ construction sites
  - **Pattern**: OOP builder with fluent API for method chaining
  - **Benefits**: Single source of truth, standardized raw_output handling, improved maintainability

**Usage examples:**
```python
from lib.results_builder import ResultsBuilder, results_for_ecm, results_for_stage1

# Basic usage with method chaining
results = (ResultsBuilder('123456789', 'ecm')
           .with_b1(50000)
           .with_b2(5000000)
           .with_curves(100, 75)
           .with_parametrization(3)
           .with_execution_time(123.45)
           .build())

# Add factors with sigmas
builder = ResultsBuilder('999', 'ecm')
builder.with_factors([('123', '12345'), ('456', '67890')])
results = builder.build()

# Stage1-only mode
results = results_for_stage1('123', b1=50000, curves=100, param=1).build()

# Add raw output incrementally
builder = ResultsBuilder('123', 'ecm')
builder.add_raw_output('Line 1')
builder.add_raw_output('Line 2')
builder.add_raw_outputs(['Line 3', 'Line 4'])
results = builder.build()  # Joins with '\n\n', truncates at 10k chars

# Mutation-heavy code pattern (for gradual migration)
builder = ResultsBuilder('123', 'ecm').with_b1(50000)
results = builder._data.copy()  # Get mutable dict for legacy code
# ... mutate results ...
```

**Key methods:**
- `with_b1(b1)`, `with_b2(b2)` - Set bounds
- `with_curves(requested, completed=0)` - Set curve counts
- `with_factors(all_factors)` - Add list of `(factor, sigma)` tuples
- `with_single_factor(factor, sigma=None)` - Add single factor
- `add_raw_output(line)`, `add_raw_outputs(lines)` - Accumulate output lines
- `with_parametrization(param)` - Set parametrization (0-3)
- `with_execution_time(seconds)` - Set execution time
- `as_stage1_only()` - Set b2=0 for stage1-only mode
- `as_two_stage()` - Mark as two-stage execution
- `as_multiprocess(workers)` - Mark as multiprocess execution
- `build(truncate_output=10000)` - Build final dict with optional truncation
- `build_no_truncate()` - Build with full raw output

**Factory functions:**
- `results_for_ecm(composite, b1, curves, param=None)` - Quick ECM results
- `results_for_stage1(composite, b1, curves, param=None)` - Stage1-only results (b2=0)

**Migration notes:**
- **DEPRECATED**: `BaseWrapper.create_base_results()` - Use ResultsBuilder for new code
- The deprecated method is maintained for backward compatibility with existing code
- For mutation-heavy code, use `builder._data.copy()` pattern during migration

### Error Handling and Timeouts
- **Subprocess timeouts**: 1 hour for ECM, 2-4 hours for YAFU operations
- **API submission retries**: Exponential backoff with configurable attempts
- **Raw output preservation**: All outputs saved to `data/outputs/` for debugging
- **Failed submission persistence**: JSON files in `data/results/` for manual retry

### File Organization
All data directories are auto-created on first use:
- **Raw outputs**: `data/outputs/` (configured via `execution.output_dir`)
  - Timestamped files with method and curve count
- **Logs**: `data/logs/ecm_client.log` (configured via `logging.file`)
  - Combined file + console logging
- **Factors found**:
  - `data/factors_found.txt` - Human-readable format with timestamps
  - `data/factors.json` - Machine-readable with all metadata
- **Residue files**: `data/residues/` (configured via `execution.residue_dir`)
  - Auto-generated when using two-stage mode without `--save-residues`
  - Filename format: `residue_<composite_hash>_<timestamp>.txt`
  - Override with `--save-residues /path/to/custom.txt`
- **Failed submissions**: `data/results/` (for retry via `resend_failed.py`)

### Code Quality Checks
When making code changes, always run both syntax and type checks:
```bash
# Basic syntax check (catches syntax errors only)
python3 -m py_compile *.py

# Type checking (catches type hint issues, undefined variables in annotations)
python3 -m mypy --ignore-missing-imports *.py

# OR use pylint for comprehensive linting
pylint *.py
```

**Important**: `py_compile` alone is insufficient - it does not validate type hints or catch undefined names in type annotations. Always use mypy or pylint for thorough validation, especially after refactoring.

## Recent Bug Fixes and Improvements

### Graceful Shutdown (2026-01)
Multi-level Ctrl+C handling for Stage 2 and CPU Stage 1 execution:
- **1st Ctrl+C**: "Completing current batch..." - Workers finish their current chunks, then submit results
- **2nd Ctrl+C**: "Stopping after current curve..." - Workers stop after completing the current curve, then submit
- **3rd Ctrl+C**: Immediate abort

**Implementation**: `lib/ecm_executor.py` uses `shutdown_level` counter and shared `stop_event` with `Stage2Executor`

**Note**: GPU Stage 1 runs all curves in a single batch, so only batch-level shutdown applies there.

### Submission Flag Changes (2026-01)
- **`ecm_wrapper.py`**: Changed from `--no-submit` to `--submit` (opt-in submission)
  - Default behavior: Results are NOT submitted to API
  - Use `--submit` flag to enable API submission
- **`ecm_client.py`**: Unchanged - always submits by default
  - Use `--no-submit` to disable submission

### Server-Side Residue Validation (2026-01)
The server now validates stage 2 completions before accepting them:
- **Factor found**: Always accepted (any number of curves)
- **No factor**: Must complete at least 75% of assigned curves
- **Invalid completions**: Residue is released back to the available pool with error message

**Implementation**: `server/app/services/residue_manager.py:complete_residue()`

This prevents buggy clients from marking residues as complete without actually processing them.

### Two-Stage ECM Improvements
- **Exit code handling**: Fixed stage 1 to treat factor discovery (exit code 8) as success, not failure
- **B2 parameter accuracy**: When factor found in stage 1, now correctly submits with `b2=0` (stage 2 never ran)
- **Timing accuracy**: Pipeline now submits combined stage1 + stage2 execution time

### Residue File Handling
- **GPU format support**: `lib/residue_manager.py` now auto-detects and handles both GPU (single-line) and CPU (multi-line) residue file formats
- **Format detection**: Checks first 5 lines for `METHOD=ECM; SIGMA=...; ...` pattern to identify GPU format

### FactorDB Integration (aliquot_wrapper.py)
- **Retry logic**: 3 automatic retries with exponential backoff (1s, 2s) for transient server errors (502, etc.)
- **Enhanced logging**: All FactorDB operations logged to `client/data/logs/ecm_client.log` with:
  - Success/failure status for each factor submission
  - Partial failure tracking (some factors succeed, others fail)
  - Retry attempt logging with countdowns
- **View logs**: `grep "FactorDB" client/data/logs/ecm_client.log`

### Aliquot Sequence Factorization
- **Primality checks**: Added Miller-Rabin primality tests after trial division AND after ECM
- **CADO-NFS failure detection**: Now properly detects when CADO crashes and stops sequence instead of submitting partial results
- **Early termination**: Avoids wasting compute on ECM/CADO when cofactor is already prime

### Pipeline Batch Processing
- **Failure handling**: No longer submits results when stage 2 fails (e.g., residue file split errors)
- **No false failures**: Fixed detection logic - `None` return from stage 2 means "no factor found" (success), not failure

### Scientific Notation Support (2025-11)
- **B1/B2 parameters**: Now accept scientific notation for easier entry of large bounds
  - Examples: `--b1 26e7` (260,000,000), `--b2 4e11` (400,000,000,000)
  - Supports: lowercase/uppercase e (26e7, 26E7), decimals (2.6e8), explicit + sign (26e+7)
  - Works in: `ecm_wrapper.py`, `yafu_wrapper.py`, `scripts/run_batch_pipeline.py`
- **Unit tests**: Comprehensive test coverage in `client/tests/test_arg_parser.py` (14 tests, 170 lines)

### Testing Status Page Improvements (2025-11)
- **Server-side pagination**: Handles 20k+ composites efficiently (200 per page default)
  - Page load: ~0.5s vs 10+ seconds previously (200 composites + 1 query vs N+1 queries)
  - Previous/Next navigation with filter state preserved
- **Server-side filtering**: Filter composites by:
  - T-level range (min/max current t-level)
  - Priority threshold (minimum priority)
  - SNFS difficulty (exact match)
- **Performance fix**: Eliminated N+1 query problem with aggregated method counts
  - Single GROUP BY query instead of N separate queries per composite
- **No auto-refresh**: Page no longer auto-refreshes (too heavy for large datasets)

### Server Dashboard Improvements
- **Group order display**: Composite details page now shows elliptic curve group order data for factors
  - Shows: Factor, Sigma, Group Order, Group Order Factorization
  - Only displayed when group order information is available (requires sigma and parametrization)
- **Deduplicated factors**: Work summary now shows unique factors sorted numerically
  - Same factor appearing in multiple attempts now shown only once
- **Multi-factor indicators**: Recent attempts tables show `[+N more]` badge when multiple factors found in one run
- **Delete button**: Admin composite details page now has delete button with confirmation dialog
- **Auto-refresh**: Admin dashboard auto-refreshes every 30 seconds

### Multi-Factor Batch Submission (2025-10-28)
**Critical bug fix**: When multiple factors were found in a single ECM run, only the first factor was being logged to the correct composite. Subsequent factors were logged to the wrong composite (the cofactor) because the server updated the composite after processing each factor.

**Solution implemented**:
- **Extended API schema** (`server/app/schemas/submit.py`): Added `FactorWithSigma` schema and `factors_found` list to `ResultsSchema`
- **Server batch processing** (`server/app/api/v1/submit.py`):
  - All factors are now validated and added to the database BEFORE any composite updates
  - Factors are divided out sequentially from a running cofactor (not from the composite record)
  - Composite is updated only ONCE after all factors are processed
  - Robust handling: skips factors that don't divide (handles composite factors gracefully)
- **Client single submission** (`client/lib/api_client.py`, `client/lib/base_wrapper.py`):
  - `build_submission_payload()` now includes all factors with their individual sigmas in `factors_found` list
  - All factors submitted in a single API call
  - Maintains backward compatibility with `factor_found` field for single-factor submissions

**Result**: Multiple factors from the same ECM run are now correctly associated with the original composite, not its cofactors.

## Important File Locations

### Server Structure
- **Main application**: `server/app/main.py` - FastAPI app setup and middleware
- **Configuration**: `server/app/config.py` - Environment settings with Pydantic
- **Database setup**: `server/app/database.py` - SQLAlchemy engine and session
- **Models**: `server/app/models/*.py` - Database table definitions
- **API schemas**: `server/app/schemas/*.py` - Request/response validation
- **API routes**: `server/app/api/v1/*.py` - Core API endpoints (submit, work, stats, factors, residues)
- **Admin routes**: `server/app/api/v1/admin/*.py` - Modular admin endpoints
  - `dashboard.py` - Admin dashboard and summary stats
  - `composites.py` - Composite upload, bulk operations, CRUD
  - `work.py` - Work assignment management
  - `projects.py` - Project organization
  - `maintenance.py` - T-level recalculation utilities
- **Services**: `server/app/services/*.py` - Business logic layer
  - `composite_manager.py` - Composite CRUD, bulk loading, updates
  - `composites.py` - Core composite operations
  - `factors.py` - Factor validation and management
  - `t_level_calculator.py` - ECM t-level calculations (excludes superseded attempts)
  - `residue_manager.py` - Residue file storage, parsing, and lifecycle management
  - `group_order.py` - Elliptic curve group order calculation using PARI/GP
- **Utilities**: `server/app/utils/*.py` - Shared utilities
  - `serializers.py` - Database model to API dict conversion
  - `query_helpers.py` - Reusable database query patterns
  - `html_helpers.py` - Template formatting and HTML escaping
- **Templates**: `server/app/templates/` - Jinja2 HTML templates
  - `base.html` - Shared CSS and layout
  - `admin/` - Admin dashboard templates
  - `public/` - Public dashboard templates
  - `components/` - Reusable UI components
- **Migrations**: `server/migrations/` - Alembic database migrations

### Security Features
- **Admin authentication**: All admin endpoints require API key via `X-Admin-Key` header
- **Timing attack protection**: Constant-time key comparison using `secrets.compare_digest()`
- **Submission validation**: Only composites already in database can receive ECM results (prevents accidental pollution)
  - Returns 404 with helpful error message for unregistered composites
  - `ecm_wrapper.py` defaults to no submission (use `--submit` to opt-in)
  - `ecm_client.py` always submits (use `--no-submit` to opt-out)
- **File upload limits**: 10 MB maximum file size on bulk upload endpoints
- **Input validation**: UTF-8 encoding validation, Pydantic schema validation
- **Error sanitization**: Generic error messages to clients, detailed logging server-side
- **SQL injection protection**: SQLAlchemy ORM with parameterized queries
- **XSS protection**: Jinja2 auto-escaping with explicit `esc()` for HTML output

### Client Structure

#### Entry Points (Two Separate Scripts by Design)
The client has two entry points for different use cases:

| Script | Purpose | Composite Source | Residue Handling |
|--------|---------|------------------|------------------|
| `ecm_client.py` | Server-coordinated work | From server API | Upload/download via server |
| `ecm_wrapper.py` | Local/manual factorization | `--composite` argument | Local files only |

**`ecm_client.py`** - Server-coordinated modes:
- Auto-work: Continuously requests work from server
- `--composite`: Target specific composite (server provides t-level info)
- `--stage1-only`: Upload residues to server
- `--stage2-only`: Download residues from server (with `--min-b1`/`--max-b1` filters)

**`ecm_wrapper.py`** - Local/manual modes:
- Requires `--composite` argument
- `--stage2-only <file>`: Process a local residue file
- `--stage1-only` with `--upload`: Save locally and optionally upload

#### Parser Architecture
Both entry points use parsers defined in `lib/arg_parser.py`:
- `create_client_parser()` - For `ecm_client.py` (server-coordinated)
- `create_ecm_parser()` - For `ecm_wrapper.py` (local/manual)
- `create_yafu_parser()` - For `yafu_wrapper.py`

Note: `--stage2-only` has different semantics in each parser:
- In `create_client_parser()`: Boolean flag (downloads from server)
- In `create_ecm_parser()`: String path to local residue file

#### Other Files
- **Configuration**: `client/client.yaml` - Binary paths and API settings
- **Base classes**: `client/lib/base_wrapper.py` - Shared wrapper functionality
- **Utilities**: `client/lib/` - Implementation modules (parsing, configuration, execution engine)
- **Batch scripts**: `client/scripts/` - Automated processing workflows

### Database Connection
- **Default URL**: `postgresql://ecm_user:ecm_password@localhost:5432/ecm_distributed`
- **Docker port**: PostgreSQL exposed on port 5434 (host) → 5432 (container)
- **Environment**: Set `DATABASE_URL` to override default connection string