"""
Gain-Cost Model Configuration Loader

This module loads calibrated Gain-Cost model parameters from gain_cost_params.json.
If the file doesn't exist, it provides default values and warns the user to run calibration.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import warnings


class GainCostConfig:
    """Gain-Cost model parameter configuration"""
    
    # Default parameters (conservative estimates)
    DEFAULT_PARAMS = {
        'migration_cost': 0.1,  # seconds per request
        'preemption_penalty': 0.05,  # seconds
        'scalability_encode': 0.80,
        'scalability_prefill': 0.90,
        'scalability_decoding': 0.75,
        'max_decode_token_budget': 2000,  # max total tokens in decode batch (capacity)
    }
    
    def __init__(self, config_path: str = None):
        """
        Initialize Gain-Cost configuration.
        
        Args:
            config_path: Path to gain_cost_params.json. If None, searches in:
                        1. Current directory
                        2. Project root
                        3. ~/.elasticmm/
        """
        self.params = self.DEFAULT_PARAMS.copy()
        self.config_loaded = False
        self.config_path = None
        
        if config_path:
            self._load_from_path(config_path)
        else:
            self._auto_discover_config()
        
        if not self.config_loaded:
            warnings.warn(
                "\n" + "="*80 + "\n"
                "⚠️  Gain-Cost parameters not found!\n"
                "Using default values which may not be optimal for your hardware.\n\n"
                "To calibrate parameters for your system, run:\n"
                "    python examples/calibrate_gain_cost.py\n\n"
                "This will generate 'gain_cost_params.json' with optimized parameters.\n"
                + "="*80,
                UserWarning
            )
    
    def _load_from_path(self, path: str) -> bool:
        """Load parameters from a specific path"""
        try:
            with open(path, 'r') as f:
                loaded_params = json.load(f)
            self.params.update(loaded_params)
            self.config_loaded = True
            self.config_path = path
            print(f"✅ Loaded Gain-Cost parameters from {path}")
            self._print_params()
            return True
        except FileNotFoundError:
            return False
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse {path}: {e}. Using default parameters.")
            return False
    
    def _auto_discover_config(self):
        """Auto-discover configuration file"""
        search_paths = [
            Path.cwd() / "gain_cost_params.json",  # Current directory
            Path(__file__).parent.parent.parent / "gain_cost_params.json",  # Project root
            Path.home() / ".elasticmm" / "gain_cost_params.json",  # User home
        ]
        
        for path in search_paths:
            if path.exists():
                self._load_from_path(str(path))
                return
    
    def _print_params(self):
        """Print loaded parameters"""
        print("Gain-Cost Model Parameters:")
        print(f"  Migration Cost:        {self.params['migration_cost']:.4f} s/request")
        print(f"  Preemption Penalty:    {self.params['preemption_penalty']:.4f} s")
        print(f"  Scalability (E/P/D):   {self.params['scalability_encode']:.2f} / "
              f"{self.params['scalability_prefill']:.2f} / {self.params['scalability_decoding']:.2f}")
        print(f"  Max Decode Budget:     {self.params['max_decode_token_budget']} tokens (batch capacity)")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value"""
        return self.params.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.params[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Return all parameters as a dictionary"""
        return self.params.copy()


# Global singleton instance
_global_config = None


def get_gain_cost_config(config_path: str = None, force_reload: bool = False) -> GainCostConfig:
    """
    Get the global Gain-Cost configuration instance.
    
    Args:
        config_path: Optional path to configuration file
        force_reload: Force reload configuration even if already loaded
    
    Returns:
        GainCostConfig instance
    """
    global _global_config
    
    if _global_config is None or force_reload:
        _global_config = GainCostConfig(config_path)
    
    return _global_config

