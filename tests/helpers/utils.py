from pathlib import Path
from typing import Any, List, Optional

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

RELATIVE_CONFIG_PATH = "../../configs"  # relative to this utils.py file


def compose_config(
    config_name: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Compose a Hydra configuration from a config file.

    Args:
        config_name: Name of the configuration file (without .yaml extension).
        overrides: Optional list of override strings (e.g., ["key=value", "key2=value2"]).

    Returns:
        Composed DictConfig object.
    """
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path=RELATIVE_CONFIG_PATH):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def get_config_stem(config_path: str) -> str:
    """
    Extract the stem (filename without suffix) from a config file path.

    Args:
        config_path: Path to a configuration file.

    Returns:
        Filename without extension.

    Example:
        >>> get_config_stem("experiment/evaluation/single_system/foo.yaml")
        "foo"
    """
    return Path(config_path).stem


def extract_test_sequence(cfg: DictConfig) -> Any:
    """
    Extract a test sequence from a configuration object.

    Looks for either a 'sequence' attribute or 'test_sequences' attribute in cfg.data.
    If test_sequences is a list, returns the first element.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Test sequence string or first element if test_sequences is a list.
    """
    seq = getattr(cfg.data, "sequence", None) or cfg.data.test_sequences
    return seq[0] if isinstance(seq, list) else seq
