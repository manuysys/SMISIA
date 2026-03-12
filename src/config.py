"""
SMISIA — Configuration Loader
Carga config.yml y provee acceso global a la configuración.
"""

import os
import yaml
from pathlib import Path

_CONFIG = None
_CONFIG_PATH = None


def get_project_root() -> Path:
    """Devuelve la raíz del proyecto (donde está config.yml)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> dict:
    """Carga y cachea la configuración desde YAML."""
    global _CONFIG, _CONFIG_PATH
    if config_path is None:
        config_path = os.environ.get(
            "SMISIA_CONFIG",
            str(get_project_root() / "config.yml"),
        )
    if _CONFIG is not None and _CONFIG_PATH == config_path:
        return _CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    _CONFIG_PATH = config_path
    return _CONFIG


def get_config(section: str = None) -> dict:
    """Obtiene la config (o una sección)."""
    cfg = load_config()
    if section:
        return cfg.get(section, {})
    return cfg
