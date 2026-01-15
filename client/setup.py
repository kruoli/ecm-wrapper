#!/usr/bin/env python3
"""
ECM Client Setup Script

Interactive setup wizard to create client.local.yaml configuration file.
Run this before using ecm_client.py for the first time.

Usage:
    python3 setup.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_input(prompt: str, default: str = "", required: bool = False) -> str:
    """Get user input with optional default value."""
    if default:
        display_prompt = f"{prompt} [{default}]: "
    else:
        display_prompt = f"{prompt}: "

    while True:
        value = input(display_prompt).strip()
        if not value:
            if default:
                return default
            elif required:
                print("  This field is required. Please enter a value.")
                continue
            else:
                return ""
        return value


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not value:
            return default
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False
        print("  Please enter 'y' or 'n'")


def detect_cpu_cores() -> int:
    """Detect number of CPU cores."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def detect_hostname() -> str:
    """Detect machine hostname."""
    try:
        return platform.node() or "my-machine"
    except Exception:
        return "my-machine"


def find_binary(name: str, common_paths: list) -> str:
    """Try to find a binary in PATH or common locations."""
    # Check PATH first
    found = shutil.which(name)
    if found:
        return found

    # Check common paths
    for path in common_paths:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    return ""


def check_gpu() -> tuple:
    """Check for NVIDIA GPU availability."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            return True, gpu_name
    except Exception:
        pass
    return False, None


def main():
    print()
    print("=" * 60)
    print("  ECM Client Setup Wizard")
    print("=" * 60)
    print()
    print("This wizard will help you create a client.local.yaml file")
    print("with your personal settings for the ECM factorization client.")
    print()

    # Check if config already exists
    config_path = Path("client.local.yaml")
    if config_path.exists():
        print("WARNING: client.local.yaml already exists!")
        if not get_yes_no("Do you want to overwrite it?", default=False):
            print("\nSetup cancelled. Your existing configuration was preserved.")
            return
        print()

    # ============================================================
    # User Information
    # ============================================================
    print("-" * 60)
    print("USER INFORMATION")
    print("-" * 60)
    print()

    username = get_input(
        "Enter your username (for tracking your contributions)",
        required=True
    )

    default_machine = detect_hostname()
    machine_name = get_input(
        "Enter a name for this machine",
        default=default_machine
    )

    # ============================================================
    # API Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("API CONFIGURATION")
    print("-" * 60)
    print()

    print("The ECM client submits results to a coordination server.")
    print("The default production server is: https://ecm.kyleaskine.com/api/v1")
    print()

    use_default_api = get_yes_no("Use the default production server?", default=True)
    if use_default_api:
        api_endpoint = "https://ecm.kyleaskine.com/api/v1"
    else:
        api_endpoint = get_input(
            "Enter API endpoint URL",
            default="http://localhost:8000/api/v1"
        )

    # ============================================================
    # GPU Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("GPU CONFIGURATION")
    print("-" * 60)
    print()

    has_gpu, gpu_name = check_gpu()

    if has_gpu:
        print(f"Detected GPU: {gpu_name}")
        gpu_enabled = get_yes_no("Enable GPU acceleration?", default=True)
    else:
        print("No NVIDIA GPU detected.")
        gpu_enabled = get_yes_no("Enable GPU anyway? (for manual setup)", default=False)

    gpu_device = 0
    if gpu_enabled:
        gpu_device_str = get_input("GPU device number", default="0")
        try:
            gpu_device = int(gpu_device_str)
        except ValueError:
            gpu_device = 0

    # ============================================================
    # ECM Binary
    # ============================================================
    print()
    print("-" * 60)
    print("ECM BINARY")
    print("-" * 60)
    print()

    # Try to find ECM binary
    ecm_paths = [
        "~/ecm",
        "~/ecm-master/ecm",
        "/usr/local/bin/ecm",
        "/usr/bin/ecm",
        "~/gmp-ecm/ecm",
    ]
    detected_ecm = find_binary("ecm", ecm_paths)

    if detected_ecm:
        print(f"Detected ECM binary: {detected_ecm}")
        use_detected = get_yes_no("Use this ECM binary?", default=True)
        if use_detected:
            ecm_path = detected_ecm
        else:
            ecm_path = get_input("Enter path to ECM binary", required=True)
    else:
        print("ECM binary not found in common locations.")
        print("You can download pre-built binaries from:")
        print("  https://ecm.kyleaskine.com/downloads/")
        print()
        ecm_path = get_input(
            "Enter path to ECM binary (or 'ecm' if in PATH)",
            default="ecm"
        )

    # ============================================================
    # Worker Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("WORKER CONFIGURATION")
    print("-" * 60)
    print()

    cpu_cores = detect_cpu_cores()
    print(f"Detected {cpu_cores} CPU cores.")

    default_workers = min(cpu_cores, 8)  # Reasonable default
    workers_str = get_input(
        "Number of parallel workers for stage 2 / multiprocess",
        default=str(default_workers)
    )
    try:
        workers = int(workers_str)
    except ValueError:
        workers = default_workers

    # ============================================================
    # Optional: YAFU Binary
    # ============================================================
    print()
    print("-" * 60)
    print("OPTIONAL: YAFU BINARY")
    print("-" * 60)
    print()

    configure_yafu = get_yes_no("Do you have YAFU installed?", default=False)
    yafu_path = None
    yafu_threads = workers

    if configure_yafu:
        yafu_paths = [
            "~/yafu/yafu",
            "~/yafu-master/yafu",
            "/usr/local/bin/yafu",
        ]
        detected_yafu = find_binary("yafu", yafu_paths)

        if detected_yafu:
            print(f"Detected YAFU binary: {detected_yafu}")
            yafu_path = get_input("Path to YAFU", default=detected_yafu)
        else:
            yafu_path = get_input("Enter path to YAFU binary")

        if yafu_path:
            yafu_threads_str = get_input("YAFU threads", default=str(workers))
            try:
                yafu_threads = int(yafu_threads_str)
            except ValueError:
                yafu_threads = workers

    # ============================================================
    # Generate Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("GENERATING CONFIGURATION")
    print("-" * 60)
    print()

    # Build the YAML content
    config_lines = [
        "# ECM Client Local Configuration",
        f"# Generated by setup.py on {platform.node()}",
        "# This file overrides settings in client.yaml",
        "",
        "api:",
        f'  endpoint: "{api_endpoint}"',
        "",
        "client:",
        f'  username: "{username}"',
        f'  cpu_name: "{machine_name}"',
        "",
        "programs:",
        "  gmp_ecm:",
    ]

    # ECM path - use quotes if it contains special characters
    if " " in ecm_path or ecm_path.startswith("~"):
        config_lines.append(f'    path: "{ecm_path}"')
    else:
        config_lines.append(f"    path: {ecm_path}")

    config_lines.extend([
        f"    gpu_enabled: {'true' if gpu_enabled else 'false'}",
        f"    gpu_device: {gpu_device}",
        f"    workers: {workers}",
    ])

    # Optional YAFU config
    if yafu_path:
        config_lines.extend([
            "",
            "  yafu:",
        ])
        if " " in yafu_path or yafu_path.startswith("~"):
            config_lines.append(f'    path: "{yafu_path}"')
        else:
            config_lines.append(f"    path: {yafu_path}")
        config_lines.append(f"    threads: {yafu_threads}")

    config_lines.append("")  # Trailing newline

    config_content = "\n".join(config_lines)

    # Show preview
    print("Configuration preview:")
    print()
    print("-" * 40)
    print(config_content)
    print("-" * 40)
    print()

    if get_yes_no("Save this configuration?", default=True):
        with open(config_path, 'w') as f:
            f.write(config_content)

        print()
        print("=" * 60)
        print("  SETUP COMPLETE!")
        print("=" * 60)
        print()
        print(f"Configuration saved to: {config_path.absolute()}")
        print()
        print("You can now run the ECM client:")
        print()
        print("  # Auto-work mode (get work from server)")
        print("  python3 ecm_client.py")
        print()
        print("  # With specific parameters")
        print("  python3 ecm_client.py --b1 11000000 --stage1-only")
        print()
        print("  # Test without submitting (ecm_wrapper.py)")
        print("  python3 ecm_wrapper.py --composite \"123456789\" --curves 10 --b1 11000")
        print()
        print("For more options, run: python3 ecm_client.py --help")
        print()
    else:
        print("\nSetup cancelled. No files were written.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
