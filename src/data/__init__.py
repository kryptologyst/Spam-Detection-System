"""Data processing modules for the spam detection system."""

from .dataset import SpamDataset, create_synthetic_dataset
from .preprocessing import TextPreprocessor, RuleBasedFeatures
from .splits import create_splits

__all__ = ["SpamDataset", "create_synthetic_dataset", "TextPreprocessor", "RuleBasedFeatures", "create_splits"]
