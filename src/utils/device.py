"""Device management utilities."""

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device_preference: Device preference ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device object.
    """
    if device_preference == "cpu":
        return torch.device("cpu")
    
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    
    if device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # Auto selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        Dictionary with device information.
    """
    info = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    
    if info["cuda"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory"] = torch.cuda.get_device_properties(0).total_memory
    
    return info
