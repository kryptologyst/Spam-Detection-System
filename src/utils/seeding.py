"""Seeding utilities for reproducible results."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_random_state() -> dict:
    """Get current random state for all libraries.
    
    Returns:
        Dictionary containing random states.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def set_random_state(state: dict) -> None:
    """Set random state from saved state.
    
    Args:
        state: Dictionary containing random states.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])
