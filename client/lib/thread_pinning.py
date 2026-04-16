"""
Thread pinning (CPU affinity) helpers.

Binding each ECM worker to a dedicated CPU core avoids cache-thrash from
cross-core migration and reduces NUMA traffic, giving a measurable stage 2
throughput uplift (reported 3-20%) on multi-socket systems.

Pinning is Linux-only: macOS and Windows lack os.sched_setaffinity. On
unsupported platforms, is_supported() returns False and the caller should
skip pinning (ideally reject --pin-threads at parse time).

Respects pre-existing affinity (taskset / cgroups): get_available_cpus()
returns exactly what os.sched_getaffinity(0) reports, so a caller restricted
to 8 of 16 cores pins its workers within that subset.
"""
import os
import sys
from typing import List


def is_supported() -> bool:
    """Return True if CPU affinity APIs are available on this platform."""
    return hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity')


def get_available_cpus() -> List[int]:
    """
    Return sorted list of CPUs this process is allowed to run on.

    Respects any pre-existing affinity restriction (taskset, cgroups, etc).
    Returns empty list on unsupported platforms.
    """
    if not is_supported():
        return []
    return sorted(os.sched_getaffinity(0))


def resolve_pin_assignments(num_workers: int) -> List[int]:
    """
    Resolve one CPU per worker from the available affinity set.

    Args:
        num_workers: Number of workers to pin.

    Returns:
        List of length num_workers, where element i is the CPU for worker i.

    Raises:
        RuntimeError: If platform unsupported or fewer available CPUs than workers.
    """
    if not is_supported():
        raise RuntimeError(
            "Thread pinning (--pin-threads) is not supported on this platform "
            f"(os.sched_setaffinity unavailable on {sys.platform})"
        )
    available = get_available_cpus()
    if num_workers > len(available):
        raise RuntimeError(
            f"Cannot pin {num_workers} workers: only {len(available)} CPU affinity "
            f"slot(s) available {available}. Reduce --workers or widen process affinity."
        )
    return available[:num_workers]


def pin_current_process(cpu: int) -> None:
    """Pin the current process to a single CPU core. No-op on unsupported platforms."""
    if is_supported():
        os.sched_setaffinity(0, {cpu})
