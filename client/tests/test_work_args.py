#!/usr/bin/env python3
"""
Tests for the WorkArgs typed args dataclass.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.work_args import WorkArgs


class TestWorkArgsDefaults:
    """A bare WorkArgs() should be safe to read from every documented field."""

    def test_default_construction(self):
        args = WorkArgs()
        # Mode flags default to False so factory dispatch falls through
        assert args.composite is None
        assert args.pm1 is False
        assert args.pp1 is False
        assert args.p1 is False
        assert args.stage1_only is False
        assert args.stage2_only is False
        assert args.adaptive is False
        assert args.standard is False

    def test_numeric_defaults_match_argparse(self):
        args = WorkArgs()
        assert args.b1 is None
        assert args.b2 is None
        assert args.curves is None
        assert args.workers is None
        assert args.progress_interval == 0
        assert args.pp1_curves == 3
        assert args.method == "ecm"
        assert args.work_type == "standard"

    def test_auto_work_default_true(self):
        # ecm_client.py treats auto-work as implicit; default reflects that
        assert WorkArgs().auto_work is True


class TestFromNamespace:
    """from_namespace() should copy known fields and drop the rest."""

    def test_basic_round_trip(self):
        ns = argparse.Namespace(
            composite="12345",
            b1=50000,
            b2=5000000,
            curves=100,
            verbose=True,
            project="testproj",
        )
        args = WorkArgs.from_namespace(ns)
        assert args.composite == "12345"
        assert args.b1 == 50000
        assert args.b2 == 5000000
        assert args.curves == 100
        assert args.verbose is True
        assert args.project == "testproj"

    def test_mode_flags(self):
        ns = argparse.Namespace(stage1_only=True, gpu=True)
        args = WorkArgs.from_namespace(ns)
        assert args.stage1_only is True
        assert args.gpu is True
        assert args.stage2_only is False  # default preserved

    def test_unknown_namespace_attrs_ignored(self):
        # Extra fields on the Namespace (e.g. --config from create_ecm_parser)
        # are silently dropped — WorkArgs is decoupled from the parser shape.
        ns = argparse.Namespace(
            composite="9999",
            config="client.yaml",  # not a WorkArgs field
            unrelated_thing=42,
        )
        args = WorkArgs.from_namespace(ns)
        assert args.composite == "9999"
        assert not hasattr(args, "config")
        assert not hasattr(args, "unrelated_thing")

    def test_missing_namespace_attrs_use_defaults(self):
        # An empty Namespace should produce a default WorkArgs
        ns = argparse.Namespace()
        args = WorkArgs.from_namespace(ns)
        assert args == WorkArgs()

    def test_none_values_preserved(self):
        # None on the Namespace should override the default if the default isn't None
        # (e.g. method has default 'ecm', but Namespace(method=None) yields None).
        # This matches argparse behavior when a flag isn't given for a `default=None` arg.
        ns = argparse.Namespace(method=None)
        args = WorkArgs.from_namespace(ns)
        assert args.method is None

    def test_p1_sweep_fields(self):
        ns = argparse.Namespace(p1=True, pp1_curves=5)
        args = WorkArgs.from_namespace(ns)
        assert args.p1 is True
        assert args.pp1_curves == 5
        assert args.pm1 is False

    def test_stage2_filters(self):
        ns = argparse.Namespace(
            stage2_only=True,
            min_b1=1_000_000,
            max_b1=100_000_000,
            min_target_tlevel=35.0,
            max_target_tlevel=50.0,
        )
        args = WorkArgs.from_namespace(ns)
        assert args.stage2_only is True
        assert args.min_b1 == 1_000_000
        assert args.max_b1 == 100_000_000
        assert args.min_target_tlevel == 35.0
        assert args.max_target_tlevel == 50.0


class TestMutability:
    """ecm_client.py mutates the args after construction (auto_work, stage1_only)."""

    def test_fields_are_mutable(self):
        args = WorkArgs()
        args.stage1_only = True
        args.adaptive = True
        args.auto_work = False
        assert args.stage1_only is True
        assert args.adaptive is True
        assert args.auto_work is False
