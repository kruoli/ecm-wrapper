# Client refactor plan

Plan from review on 2026-04-27. Items are ordered by ROI (highest first within
each phase). Tick items as they land.

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Mechanical | **complete** | All items done 2026-04-27 |
| Phase 2 — Typed surface | **complete** | Items 3 + 4 done 2026-04-28/29 |
| Phase 3 — Architectural | in progress | Item 8 done 2026-06-01; 2, 5, 6, 7 pending |

---

## Phase 1 — Mechanical, low-risk

No behavior change. Each can ship independently.

- [x] **1. Split `lib/work_modes.py` (2054 lines) into a package.** *(done 2026-04-27)*
  Six WorkMode subclasses now live in their own modules under
  `lib/work_modes/`. Public symbols re-exported from `__init__.py` so
  existing imports (`from lib.work_modes import …`) keep working unchanged.
  Final layout:
  ```
  lib/work_modes/
    __init__.py          # re-exports + get_work_mode() factory
    base.py              # WorkMode + WorkLoopContext + MAX_CONSECUTIVE_FAILURES
    stage1_producer.py   # GPU producer
    stage2_consumer.py   # CPU consumer
    p1_sweep.py          # PM1/PP1 sweep
    standard.py          # StandardAutoWorkMode
    composite_target.py  # CompositeTargetMode (extends StandardAutoWorkMode)
    adaptive.py          # AdaptiveCPUMode
  ```
  Verification: 25/25 tests in `tests/test_work_modes.py` pass; full suite
  green except for the unrelated pre-existing `test_raw_output_truncation`.
  CompositeTargetMode is now its own module (originally lived inside
  `standard.py` per the initial plan, but kept separate for symmetry with
  the other modes).

- [x] **9. Collapse duplicate YAFU parsers.** *(done 2026-04-27)*
  `parse_yafu_ecm_output` and `parse_yafu_auto_factors` now delegate to a
  shared `_parse_yafu_factor_section(output, source_label)` private helper.
  Public function names kept for API stability — yafu_wrapper.py and the test
  files import them unchanged. Net: ~80 lines collapsed to ~50, with the
  identical-behavior guarantee now structural rather than coincidental.

- [x] **10. Single B2 dictionary parse.** *(done 2026-04-27)*
  `load_b2_dictionary` in `arg_parser.py` now returns `(b2_dict, k_dict)` from
  a single file read. The inline second-pass parser in `work_modes/base.py`
  (was `work_modes.py:417–444`) is gone. `ecm_wrapper.py` (manual mode) keeps
  current behavior by unpacking and discarding the k dict — k is only consumed
  by the auto-work loop.

  **Signature change:** `load_b2_dictionary(filepath) -> tuple[Dict, Dict]`
  rather than `Dict`. No callers outside the two updated sites; no tests
  exercised it directly.

- [x] **11. Fix dead code in `TLevelConfig.__post_init__`.** *(done 2026-04-27)*
  The auto-set of `parametrization=3` for two-stage mode was unreachable —
  it sat after a `return` in `get_b2_for_b1`. Moved into `__post_init__`.
  An explicit non-default `parametrization` argument still wins (the bump
  only fires when `parametrization == 1`, the dataclass default).

  Regression tests added in `tests/test_integration.py::TestECMConfigIntegration`:
  - default parametrization stays 1
  - `use_two_stage=True` auto-bumps to 3
  - explicit `parametrization=2` is preserved even with `use_two_stage=True`

  **Workarounds left in place** (intentional, low-risk): `ecm_executor.py`
  still has `effective_param = 3 if config.use_two_stage else config.parametrization`
  at lines 429/450, plus a dead `if config.use_two_stage:` branch at line 378
  inside the CPU-only loop. They're now provably redundant (the early return
  at line 286 already routes two-stage to `_run_tlevel_pipelined`, and the
  dataclass auto-set means `config.parametrization == 3` anyway). I left them
  for a separate cleanup item — they're dead but documented, and removing
  them is a separate audit of `run_tlevel_v2`'s control flow.

- [x] **12. Add `stage2_max_b1` to `AppConfig.to_dict()`.** *(done 2026-04-27)*
  One-liner: added `'stage2_max_b1': self.programs.gmp_ecm.stage2_max_b1,`
  to the `gmp_ecm` block in `AppConfig.to_dict()`. Smoke-tested round-trip
  on the live `client.yaml`; field now survives.

- [x] **13. Renamed `lib/typed_config.py:TLevelConfig` to `TLevelBinaryConfig`.**
  *(done 2026-04-27)* Resolves the naming clash with the more prominent
  `lib/ecm_config.py:TLevelConfig` (ECM execution targeting). The renamed
  class only has internal references — no other module imported it under
  the old name.

---

## Phase 2 — Typed surface

These depend on Phase 1 being landed. Decide between 3a and 3b before starting 4.

- [x] **3. Finish the `typed_config` migration.** *(done 2026-04-28)*
  Decision: option **3a (finish)**. Endpoint state: `BaseWrapper.config` (the
  dict) gets removed; `BaseWrapper.typed_config` becomes the single source.
  47 production call sites to migrate.

  **Sub-step 3.1 — schema additions** *(done 2026-04-27)*: extended
  `typed_config.py` to cover keys the audit found missing:
  - `GMPECMConfig.max_batch: Optional[int] = None`
  - `GMPECMConfig.gpu: GPUConfig` (new sub-dataclass, `curves_per_batch: int = 1000`)
  - `ExecutionConfig.queue_dir: str = "data/queue"`
  - Bonus fix: `_parse_gmp_ecm` now actually reads `stage2_max_b1` from YAML
    (was previously dropped on the read side — sibling bug to item 12).
  - 13 new round-trip tests in `tests/test_typed_config.py`.

  **Sub-step 3.2 — consumer migration** *(done 2026-04-28)*: 47 sites across
  `lib/arg_parser.py`, `lib/base_wrapper.py`, `lib/work_modes/*.py`,
  `lib/ecm_executor.py`, `lib/execution_engine.py`, `lib/stage2_executor.py`,
  `lib/ecm_modes.py`, `cado_wrapper.py`, `yafu_wrapper.py`,
  `aliquot_wrapper.py`, `ecm_wrapper.py`, `ecm_client.py`, and
  `scripts/run_batch_pipeline.py`. Migrate in dependency order
  (helpers first, then leaves, then entry points).

  **Sub-step 3.3 — remove `BaseWrapper.config`** *(done 2026-04-28)*:
  `self.config` (dict) and the lazy `_typed_config` property are gone from
  `BaseWrapper`. `__init__` now eagerly loads `self.typed_config` via
  `TypedConfigLoader().load(config_path)` and never materializes the raw
  dict on the wrapper. All call sites (47) read `self.typed_config.*`.
  Dead `self.config = {...}` block in `tests/test_work_modes.py::MockWrapper`
  was removed (it was never read). `ConfigManager` itself is retained —
  `TypedConfigLoader` still uses `ConfigManager.load_config` internally to
  perform the YAML+local deep merge before parsing into typed objects;
  that's an implementation detail, not a public surface. Verification:
  431 passed, 8 skipped across the full test suite.

  **Watch-outs surfaced by the audit:**
  - Mutation site at `aliquot_wrapper.py:992` writes
    `wrapper.config['logging']['log_factors_found'] = False`. After migration
    this becomes a typed attribute mutation; observable behavior changes
    slightly because today the dict and typed_config are independent objects
    that can drift, and after migration there's only one source.
  - The flat `gpu_enabled`/`gpu_device`/`gpu_curves` fields on `GMPECMConfig`
    coexist with the new nested `gpu` sub-dataclass. Awkward but matches the
    YAML/code reality. Consolidating would break user-facing yamls and is
    deferred.

- [x] **4. Typed `WorkArgs` dataclass.** *(done 2026-04-29)*
  New `lib/work_args.py` defines a flat `WorkArgs` dataclass covering every
  flag `create_client_parser()` exposes plus the runtime-set `auto_work`.
  `WorkArgs.from_namespace(ns)` copies known fields and silently drops
  unknown ones, so the parser shape and the typed surface stay decoupled.

  `WorkLoopContext.args` is now `WorkArgs` instead of `argparse.Namespace`.
  `ecm_client.py` converts once at the entry point (after the
  `auto_work`/interactive-mode mutations). All 41 `getattr(self.args, ...)`
  sites in `work_modes/*.py` collapse to direct attribute access — future
  parser changes become real `AttributeError`s.

  **Other surfaces touched:**
  - `lib/work_helpers.py`: `request_ecm_work` / `request_p1_work` typed as
    `WorkArgs`; the cluster of `args.X if hasattr(args, 'X') else None`
    lines is gone.
  - `lib/arg_parser.py`: `resolve_pin_threads` / `resolve_worker_count` /
    `resolve_gpu_settings` accept `ArgsLike = Union[Namespace, WorkArgs]`
    via duck typing; defensive `getattr` defaults removed (both parsers
    expose the relevant fields).
  - `lib/ecm_arg_helpers.py`: same treatment for `parse_sigma_arg` /
    `resolve_param`.

  Decided **not** to do per-mode subclasses — every work mode reads from the
  same client parser, and the field set is small enough that one flat
  dataclass is the cleanest contract.

  **Out of scope (deferred):**
  - `validate_ecm_args` in `arg_parser.py` still has one `getattr` for
    `b2_multiplier`; it's only called from `ecm_wrapper.py` (manual mode,
    raw Namespace). Tracked under the "polish" list — per-mode validators
    would clean it up.
  - `ecm_wrapper.py` and `lib/ecm_modes.py` keep argparse.Namespace; they
    don't go through the work-loop and have their own resolution flow
    (`ResolvedParams`).

  **Verification:** 442 passed, 8 skipped. New `tests/test_work_args.py`
  (11 tests) covers defaults, `from_namespace` round-trip, unknown-attr
  drop, and post-construction mutability (the auto_work/stage1_only
  pattern in `ecm_client.py`).

---

## Phase 3 — Architectural

Higher reward, more risk. Run the test suite throughout.

- [ ] **2. `AdaptiveCPUMode` should compose, not duplicate.** Currently
  reimplements stage 2 download/checksum/execute/abandon/cleanup line-by-line
  parallel with `Stage2ConsumerMode`. Concrete duplications:
  - `_compute_file_checksum` at work_modes.py:874 and :1712
  - `_cleanup_local_residue` ↔ `_cleanup_s2_residue`
  - `_complete_stage2` retraces `Stage2ConsumerMode.complete_work`
  - `if self._current_mode == 'stage2'` branching in every method = hand-rolled
    polymorphism

  Refactor: `AdaptiveCPUMode` becomes a dispatcher holding inner `Stage2ConsumerMode`
  and `StandardAutoWorkMode`/`MultiprocessECMMode` instances and forwarding
  lifecycle calls. Removes ~250 duplicated lines.

- [ ] **5. Unify the signal-handling layer.** Three parallel SIGINT handler
  installations all manipulate `wrapper.shutdown_level` /
  `stop_event` / `graceful_shutdown_requested`:
  - `WorkMode._setup_signal_handler` (work_modes.py:324) — 3 levels
  - `Stage1ProducerMode._setup_signal_handler` (work_modes.py:581) — 2 levels (GPU)
  - `CompositeExecutionEngine._install_graceful_handler` (execution_engine.py:493) — 3 levels

  The work-loop comment at work_modes.py:472 admits the systems aren't fully
  synchronized. Single `ShutdownController` class on `ECMWrapper` with
  configurable level count and explicit `install()/restore()` lifecycle.

- [ ] **6. Extract `PipelinedExecutor` from `run_pipelined()`** (execution_engine.py:528–977).
  400-line method with two huge nested closures (`gpu_producer`, `cpu_consumer`)
  using `nonlocal` to mutate result state. Stage 2 submission inside the
  closure also duplicates work_modes.py logic. Extract to a class with proper
  methods + a shared `BatchSubmitter` helper.

- [ ] **7. Consolidate stage 2 submission dict construction.** Built in 3+ places:
  - `WorkMode._submit_stage2_results` (work_modes.py:255)
  - `Stage2ConsumerMode.submit_results` (work_modes.py:1020)
  - `AdaptiveCPUMode._submit_stage2` (work_modes.py:1932)
  - The pipelined `cpu_consumer` closure (execution_engine.py:842–862)

  `ResultsBuilder` already exists for stage 1. Add `results_for_stage2()` /
  `results_for_multiprocess()` factories.

- [x] **8. Parameterize `_fully_factor_*` runner.** *(done 2026-06-01)*
  Shared body extracted to `_fully_factor_with_runner(factor, runner,
  max_ecm_attempts, quiet)` (ecm_executor.py). Both `_fully_factor_found_result`
  and `_fully_factor_composite` are now thin wrappers (~15 lines each) that
  define their runner closure and delegate. Recursion happens via
  `_fully_factor_with_runner` reusing the same runner, which preserves the
  no-run_ecm_v2 invariant of the composite path. Public signatures unchanged
  (including the unused-but-passed-through `quiet` parameter) so external
  callers (`execution_engine.py`, `result_processor.py`) and tests
  (`test_composite_factor_splitting.py`, `test_result_processor.py`) need no
  changes. ecm_executor.py: 1193 → 1120 lines (-73). All 452 tests pass.

---

## Lower priority / polish (track but don't schedule)

- `validate_ecm_args` (arg_parser.py:306) is 142 lines of `hasattr` + branching
  errors[]. Per-mode validators would be cleaner.
- `ResultsBuilder.as_two_stage` / `.as_multiprocess` / `.as_stage2_workers`
  may be dead — grep for `results['two_stage']`, `results['multiprocess']`,
  `results['workers']` consumers and delete unused flags.
- `get_ecm_work` and `get_p1_work` (api_client.py:309 and :445) share shape;
  a `_request_work(endpoint, params)` helper saves ~50 lines.
- Inconsistent submission API: `submit_stage1_complete_workflow`,
  `wrapper.submit_result`, `_submit_stage2_results`, `_submit_ecm_results`.
  Pick one shape and align.

---

## Deliberately out of scope

- **`aliquot_wrapper.py` (1173 lines)** — application using the wrapper toolkit,
  not core ECM coordination. Class is well-bounded. Leave alone unless actively
  expanding.
- **`api_client.py` (872 lines)** — flat list of API methods, not deeply nested.
  Easy to navigate as-is.
- **`base_wrapper.py`** — dense but coherent; `SubmissionResult` +
  `submit_payload_to_endpoints` handles multi-endpoint cleanly.
- **`ResultsBuilder`** — well-structured; just needs the stage 2 / multiprocess
  factories (item 7 above).
