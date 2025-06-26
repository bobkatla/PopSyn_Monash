"""Utils for processing the yaml files"""
import yaml
from pathlib import Path
from typing import Dict, Any


def handle_yaml_abs_path(yaml_path: Path, target_path: Path) -> Path:
    """Handle the absolute path of the yaml file."""
    if not yaml_path.is_absolute():
        yaml_path = target_path.parent / yaml_path
    return yaml_path
