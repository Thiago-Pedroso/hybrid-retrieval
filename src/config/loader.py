"""
Configuration loader with support for YAML/JSON files and CLI overrides.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from .schema import ExperimentConfig
from .defaults import get_default_config


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    if not _HAS_YAML:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: Optional[str | Path] = None) -> ExperimentConfig:
    """Load experiment configuration from file.
    
    Args:
        config_path: Path to YAML or JSON config file. If None, returns default config.
        
    Returns:
        Validated ExperimentConfig object
    """
    if config_path is None:
        return ExperimentConfig(**get_default_config())
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Load based on extension
    if path.suffix.lower() in (".yaml", ".yml"):
        config_dict = load_yaml(path)
    elif path.suffix.lower() == ".json":
        config_dict = load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json")
    
    # Merge with defaults (config_dict takes precedence)
    defaults = get_default_config()
    merged = _deep_merge(defaults, config_dict)
    
    return ExperimentConfig(**merged)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    return _deep_merge(base, override)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def apply_cli_overrides(config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply CLI argument overrides to configuration.
    
    Args:
        config: Base configuration dictionary
        cli_args: CLI arguments to override with
        
    Returns:
        Updated configuration
    """
    # Simple dot-notation override support
    # e.g., {"dataset.name": "fiqa"} -> config["dataset"]["name"] = "fiqa"
    updated = config.copy()
    
    for key, value in cli_args.items():
        if "." in key:
            parts = key.split(".")
            target = updated
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            updated[key] = value
    
    return updated

