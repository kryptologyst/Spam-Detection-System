"""Tests for model implementations."""

import pytest
import numpy as np
from src.models import BaselineModel, TransformerModel, EnsembleModel
from src.data import create_synthetic_dataset


def test_baseline_model():
    """Test baseline model functionality."""
    model = BaselineModel(model_type="naive_bayes")
    
    # Create simple test data
    X = np.random.rand(100, 10)
    y = ['spam' if i % 3 == 0 else 'ham' for i in range(100)]
    
    # Fit model
    model.fit(X, y)
    
    # Test prediction
    X_test = np.random.rand(10, 10)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    assert len(predictions) == 10
    assert probabilities.shape == (10, 2)
    assert all(pred in ['spam', 'ham'] for pred in predictions)


def test_baseline_model_evaluation():
    """Test baseline model evaluation."""
    model = BaselineModel(model_type="naive_bayes")
    
    # Create test data
    X = np.random.rand(100, 10)
    y = ['spam' if i % 3 == 0 else 'ham' for i in range(100)]
    
    # Fit and evaluate
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, (int, float)))


def test_transformer_model():
    """Test transformer model functionality."""
    # Skip if transformers not available
    try:
        import transformers
    except ImportError:
        pytest.skip("Transformers not available")
    
    model = TransformerModel(
        model_name="distilbert-base-uncased",
        max_length=128,
        batch_size=4,
        num_epochs=1,
        device="cpu"
    )
    
    # Create test data
    texts = ["Hello world", "Free money now!", "Meeting at 3 PM", "Click here for prize"]
    labels = ["ham", "spam", "ham", "spam"]
    
    # Fit model
    model.fit(texts, labels)
    
    # Test prediction
    test_texts = ["Hello", "Free money"]
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    assert len(predictions) == 2
    assert probabilities.shape == (2, 2)
    assert all(pred in ['spam', 'ham'] for pred in predictions)


def test_ensemble_model():
    """Test ensemble model functionality."""
    # Create simple baseline models
    baseline1 = BaselineModel(model_type="naive_bayes")
    baseline2 = BaselineModel(model_type="logistic_regression")
    
    # Create ensemble
    ensemble = EnsembleModel(
        models=[baseline1, baseline2],
        weights=[0.5, 0.5]
    )
    
    # Create test data
    X = np.random.rand(100, 10)
    y = ['spam' if i % 3 == 0 else 'ham' for i in range(100)]
    
    # Fit ensemble
    ensemble.fit(X, y)
    
    # Test prediction
    X_test = np.random.rand(10, 10)
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    assert len(predictions) == 10
    assert probabilities.shape == (10, 2)
    assert all(pred in ['spam', 'ham'] for pred in predictions)


def test_model_save_load():
    """Test model saving and loading."""
    model = BaselineModel(model_type="naive_bayes")
    
    # Create test data
    X = np.random.rand(100, 10)
    y = ['spam' if i % 3 == 0 else 'ham' for i in range(100)]
    
    # Fit model
    model.fit(X, y)
    
    # Save model
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model.save(f.name)
        
        # Load model
        loaded_model = BaselineModel.load(f.name)
        
        # Test that loaded model works
        X_test = np.random.rand(5, 10)
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        
        assert np.array_equal(original_pred, loaded_pred)
