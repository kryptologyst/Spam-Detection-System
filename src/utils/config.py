"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import OmegaConf


class Config:
    """Configuration manager for the spam detection system."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return OmegaConf.create(config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return OmegaConf.select(self._config, key, default=default)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._config = OmegaConf.merge(self._config, updates)

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            path: Path to save configuration. If None, saves to original path.
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(self._config), f, default_flow_style=False)

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.get("data.dir", "data"))

    @property
    def model_dir(self) -> Path:
        """Get model directory path."""
        return Path(self.get("model.dir", "models"))

    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        return Path(self.get("logging.dir", "logs"))

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.update({key: value})
