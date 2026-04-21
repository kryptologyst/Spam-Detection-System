"""Tests for data processing modules."""

import pytest
import numpy as np
from src.data import create_synthetic_dataset, SpamDataset, create_splits
from src.data.preprocessing import TextPreprocessor, RuleBasedFeatures, FeatureExtractor


def test_create_synthetic_dataset():
    """Test synthetic dataset creation."""
    dataset = create_synthetic_dataset(n_samples=100, seed=42)
    
    assert len(dataset.texts) == 100
    assert len(dataset.labels) == 100
    assert all(label in ['spam', 'ham'] for label in dataset.labels)
    assert all(isinstance(text, str) for text in dataset.texts)


def test_spam_dataset():
    """Test SpamDataset class."""
    texts = ["Hello world", "Free money now!"]
    labels = ["ham", "spam"]
    
    dataset = SpamDataset(texts, labels)
    
    assert len(dataset.texts) == 2
    assert len(dataset.labels) == 2
    assert dataset.texts == texts
    assert dataset.labels == labels


def test_dataset_anonymization():
    """Test PII anonymization."""
    texts = ["Contact me at john@example.com or call 555-1234"]
    labels = ["ham"]
    
    dataset = SpamDataset(texts, labels, anonymize=True)
    
    # Check that PII is anonymized
    assert "john@example.com" not in dataset.texts[0]
    assert "555-1234" not in dataset.texts[0]
    assert "email_" in dataset.texts[0]
    assert "phone_" in dataset.texts[0]


def test_create_splits():
    """Test dataset splitting."""
    dataset = create_synthetic_dataset(n_samples=100, seed=42)
    train, val, test = create_splits(dataset, seed=42)
    
    assert len(train.texts) + len(val.texts) + len(test.texts) == 100
    assert len(train.texts) > 0
    assert len(val.texts) > 0
    assert len(test.texts) > 0


def test_text_preprocessor():
    """Test text preprocessing."""
    preprocessor = TextPreprocessor()
    
    text = "Hello WORLD! This is a test."
    processed = preprocessor.preprocess(text)
    
    assert isinstance(processed, str)
    assert "world" in processed.lower()
    assert "!" not in processed  # Punctuation removed


def test_rule_based_features():
    """Test rule-based feature extraction."""
    extractor = RuleBasedFeatures()
    
    text = "Free money! Click here: https://example.com"
    features = extractor.extract_features(text)
    
    assert isinstance(features, dict)
    assert 'char_count' in features
    assert 'word_count' in features
    assert 'has_urls' in features
    assert features['has_urls'] == 1.0  # URL detected


def test_feature_extractor():
    """Test combined feature extraction."""
    extractor = FeatureExtractor()
    
    texts = ["Hello world", "Free money now!"]
    
    # Fit and transform
    tfidf_features, rule_features = extractor.fit_transform(texts)
    
    assert tfidf_features.shape[0] == 2
    assert rule_features.shape[0] == 2
    assert tfidf_features.shape[1] > 0
    assert rule_features.shape[1] > 0
