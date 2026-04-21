#!/usr/bin/env python3
"""Simple test script to verify the spam detection system works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data import create_synthetic_dataset, SpamDataset
        print("✓ Data modules imported successfully")
    except Exception as e:
        print(f"✗ Data import failed: {e}")
        return False
    
    try:
        from models import BaselineModel
        print("✓ Model modules imported successfully")
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False
    
    try:
        from eval import EvaluationMetrics
        print("✓ Evaluation modules imported successfully")
    except Exception as e:
        print(f"✗ Evaluation import failed: {e}")
        return False
    
    try:
        from utils import Config, set_seed
        print("✓ Utility modules imported successfully")
    except Exception as e:
        print(f"✗ Utility import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic system functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from data import create_synthetic_dataset
        from models import BaselineModel
        from eval import EvaluationMetrics
        from utils import set_seed
        import numpy as np
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create small dataset
        print("Creating synthetic dataset...")
        dataset = create_synthetic_dataset(n_samples=50, seed=42)
        print(f"✓ Created dataset with {len(dataset.texts)} samples")
        
        # Test baseline model
        print("Testing baseline model...")
        model = BaselineModel(model_type="naive_bayes")
        
        # Create simple features for testing
        X = np.random.rand(len(dataset.texts), 10)
        y = dataset.labels
        
        # Fit model
        model.fit(X, y)
        print("✓ Model fitted successfully")
        
        # Make predictions
        predictions = model.predict(X[:5])
        probabilities = model.predict_proba(X[:5])
        print(f"✓ Made predictions: {predictions}")
        
        # Test evaluation
        evaluator = EvaluationMetrics()
        metrics = evaluator.compute_metrics(y[:5], predictions, probabilities)
        print(f"✓ Evaluation completed: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🛡️ Spam Detection System - Basic Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed.")
        sys.exit(1)
    
    print("\n✅ All tests passed! The system is working correctly.")
    print("\nNext steps:")
    print("1. Run 'python scripts/train.py' to train models")
    print("2. Run 'streamlit run demo/app.py' to launch the demo")
    print("3. Check the README.md for more information")

if __name__ == "__main__":
    main()
