"""File I/O utilities for JSON persistence."""
import json
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_json(file_path: Path, data: Any, ensure_dir: bool = True) -> bool:
    """
    Save data to JSON file with error handling.

    Args:
        file_path: Path to save JSON file
        data: Data to serialize
        ensure_dir: If True, create parent directories

    Returns:
        True if successful, False otherwise
    """
    try:
        if ensure_dir:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def load_json(file_path: Path, default: Optional[Any] = None) -> Any:
    """
    Load data from JSON file with error handling.

    Args:
        file_path: Path to JSON file
        default: Value to return if file doesn't exist or fails to load

    Returns:
        Loaded data or default value
    """
    try:
        if not file_path.exists():
            return default

        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return default
