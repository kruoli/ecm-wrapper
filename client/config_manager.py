"""
Configuration Manager Utility

Provides unified configuration loading with deep merge support for
client.yaml and client.local.yaml files.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manage configuration loading with automatic local overrides.

    This utility consolidates configuration loading patterns, handling:
    - Loading base configuration from client.yaml
    - Auto-detecting and merging client.local.yaml overrides
    - Deep merging nested dictionaries
    """

    def __init__(self):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(f"{__name__}.ConfigManager")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with automatic local overrides.

        This method:
        1. Loads the base configuration from config_path (typically client.yaml)
        2. Looks for a corresponding .local.yaml file in the same directory
        3. If found, deep merges the local config into the base config
        4. Returns the merged configuration

        Args:
            config_path: Path to base configuration file (e.g., 'client.yaml')

        Returns:
            Merged configuration dictionary

        Raises:
            FileNotFoundError: If base config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_file = Path(config_path)

        # Validate base config exists
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        # Load base configuration
        self.logger.debug(f"Loading base configuration from: {config_path}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not config:
            self.logger.warning(f"Configuration file is empty: {config_path}")
            config = {}

        # Look for local overrides
        local_config_path = self._get_local_config_path(config_file)

        if local_config_path.exists():
            self.logger.info(
                f"Loading local configuration overrides from: {local_config_path}"
            )
            try:
                with open(local_config_path, 'r', encoding='utf-8') as f:
                    local_config = yaml.safe_load(f)

                if local_config:
                    # Deep merge local config into base config
                    config = self.deep_merge(config, local_config)
                    self.logger.debug(
                        f"Successfully merged local configuration from {local_config_path.name}"
                    )
                else:
                    self.logger.warning(
                        f"Local configuration file is empty: {local_config_path}"
                    )

            except yaml.YAMLError as e:
                self.logger.error(
                    f"Failed to parse local configuration {local_config_path}: {e}"
                )
                # Continue with base config only
            except Exception as e:
                self.logger.error(
                    f"Error loading local configuration {local_config_path}: {e}"
                )
                # Continue with base config only
        else:
            self.logger.debug(
                f"No local configuration file found at {local_config_path}"
            )

        return config

    def _get_local_config_path(self, base_config_path: Path) -> Path:
        """
        Determine the path for local configuration overrides.

        For a base config like 'client.yaml', returns 'client.local.yaml'.
        For a base config like 'config/app.yaml', returns 'config/app.local.yaml'.

        Args:
            base_config_path: Path object for base configuration

        Returns:
            Path object for local configuration file
        """
        # Replace extension: client.yaml -> client.local.yaml
        stem = base_config_path.stem  # 'client'
        parent = base_config_path.parent  # directory

        local_name = f"{stem}.local.yaml"
        return parent / local_name

    def deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge override dictionary into base dictionary.

        This method recursively merges nested dictionaries. For non-dict values,
        the override value replaces the base value.

        Example:
            base = {'a': {'b': 1, 'c': 2}, 'd': 3}
            override = {'a': {'b': 99}, 'e': 4}
            result = {'a': {'b': 99, 'c': 2}, 'd': 3, 'e': 4}

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary (new dict, inputs unchanged)
        """
        # Create a copy to avoid mutating the original
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Both values are dicts - recursively merge
                result[key] = self.deep_merge(result[key], value)
            else:
                # Override value replaces base value
                result[key] = value

        return result

    def validate_config_structure(
        self, config: Dict[str, Any], required_keys: Optional[list] = None
    ) -> bool:
        """
        Validate that configuration has required top-level keys.

        Args:
            config: Configuration dictionary to validate
            required_keys: List of required top-level keys (default: standard keys)

        Returns:
            True if all required keys are present, False otherwise
        """
        if required_keys is None:
            # Standard required keys for ECM client configuration
            required_keys = ['client', 'api', 'programs']

        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            self.logger.error(
                f"Configuration is missing required keys: {', '.join(missing_keys)}"
            )
            return False

        return True

    def get_nested_value(
        self, config: Dict[str, Any], key_path: str, default: Any = None
    ) -> Any:
        """
        Get a nested configuration value using dot notation.

        Example:
            config = {'api': {'endpoint': 'http://localhost:8000'}}
            value = get_nested_value(config, 'api.endpoint')
            # Returns: 'http://localhost:8000'

        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to nested key (e.g., 'api.timeout')
            default: Default value if key path doesn't exist

        Returns:
            Value at key_path, or default if not found
        """
        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value
