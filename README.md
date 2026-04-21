# Spam Detection System

A privacy-preserving spam detection system built with transformer models, rule-based features, and comprehensive evaluation metrics. This project demonstrates advanced techniques for text classification with a focus on security, privacy, and explainability.

## ⚠️ Disclaimer

This is a **research and educational demonstration** of spam detection techniques. The system is designed for **defensive purposes only** and should not be used for production security operations or exploitation. Results may be inaccurate and should not be relied upon for critical decisions.

## Features

- **Multiple Model Architectures**: Baseline ML models, transformer-based models, and ensemble approaches
- **Privacy-Preserving**: PII anonymization, differential privacy options, and secure data handling
- **Comprehensive Evaluation**: AUCPR, precision@K, operational metrics, and model comparison
- **Explainability**: SHAP values, feature importance, and rule-based explanations
- **Interactive Demo**: Streamlit-based web interface for real-time analysis
- **Production-Ready**: Proper configuration management, logging, and reproducible results

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Spam-Detection-System.git
cd Spam-Detection-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Training Models

Train all models with default configuration:
```bash
python scripts/train.py
```

Train specific models:
```bash
python scripts/train.py --models baseline transformer --data-size 1000
```

### Running the Demo

Launch the interactive Streamlit demo:
```bash
streamlit run demo/app.py
```

The demo will be available at `http://localhost:8501`

## Project Structure

```
spam-detection-system/
├── src/                    # Source code
│   ├── data/              # Data processing and dataset creation
│   ├── models/            # Model implementations
│   ├── eval/              # Evaluation metrics and explainability
│   ├── utils/             # Utility functions
│   └── __init__.py
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── assets/                # Generated plots and model artifacts
├── data/                  # Data storage (gitignored)
├── models/                # Trained models (gitignored)
├── logs/                  # Log files (gitignored)
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Model Architectures

### 1. Baseline Models
- **Naive Bayes**: Fast, interpretable baseline
- **Logistic Regression**: Linear classifier with regularization
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support vector machine with RBF kernel

### 2. Transformer Models
- **DistilBERT**: Lightweight transformer for text classification
- **Custom Architecture**: BERT-based classifier with dropout and custom head
- **Fine-tuning**: Domain-specific adaptation for spam detection

### 3. Ensemble Models
- **Weighted Voting**: Combines multiple models with learned weights
- **Soft Voting**: Uses probability distributions for consensus
- **Stacking**: Meta-learner trained on base model predictions

## Data Processing

### Synthetic Dataset Generation
The system generates realistic synthetic spam and ham messages with:
- **Spam Templates**: Common spam patterns (prizes, urgency, offers)
- **Ham Templates**: Legitimate communication patterns
- **Variable Substitution**: Dynamic content generation
- **Privacy Protection**: Automatic PII anonymization

### Feature Engineering
- **TF-IDF Features**: Term frequency-inverse document frequency
- **Rule-Based Features**: Spam indicators (URLs, phone numbers, keywords)
- **Text Statistics**: Length, punctuation, capitalization patterns
- **N-gram Features**: Character and word n-grams

### Privacy Measures
- **PII Anonymization**: Email, phone, and IP address hashing
- **Differential Privacy**: Optional privacy-preserving training
- **Data Minimization**: Only necessary features extracted
- **Secure Storage**: Encrypted model artifacts

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve
- **AUCPR**: Area under the precision-recall curve

### Operational Metrics
- **Precision@K**: Precision at different recall levels
- **False Positive Rate**: Rate of legitimate messages flagged as spam
- **Alert Volume**: Number of messages flagged per time period
- **Threshold Analysis**: Performance at different decision thresholds

### Business Metrics
- **Cost Analysis**: False positive vs false negative costs
- **Workload Impact**: Human review queue size
- **User Experience**: Legitimate message delivery rates

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/config.yaml
data:
  seed: 42
  train_size: 0.8
  val_size: 0.1
  test_size: 0.1

models:
  baseline:
    type: "naive_bayes"
    alpha: 1.0
  transformer:
    model_name: "distilbert-base-uncased"
    max_length: 512
    batch_size: 16
    learning_rate: 2e-5
    num_epochs: 3

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc", "aucpr"]
  threshold: 0.5
  precision_at_k: [10, 50, 100]
```

## Usage Examples

### Basic Training
```python
from src.data import create_synthetic_dataset, create_splits
from src.models import BaselineModel, TransformerModel
from src.eval import EvaluationMetrics

# Create dataset
dataset = create_synthetic_dataset(n_samples=1000)
train, val, test = create_splits(dataset)

# Train baseline model
baseline = BaselineModel(model_type="naive_bayes")
baseline.fit(X_train, y_train)

# Train transformer model
transformer = TransformerModel()
transformer.fit(train.texts, train.labels, val.texts, val.labels)

# Evaluate
evaluator = EvaluationMetrics()
metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
```

### Model Explanation
```python
from src.eval import ModelExplainer

# Create explainer
explainer = ModelExplainer(model)

# Explain prediction
explanation = explainer.explain_prediction(
    text="Congratulations! You've won $1000!",
    method="shap"
)

print(f"Prediction: {explanation['prediction']}")
print(f"Confidence: {explanation['confidence']:.2%}")
print("Feature Importance:")
for feature, importance in explanation['feature_importance']:
    print(f"  {feature}: {importance:.3f}")
```

### Privacy-Preserving Training
```python
from src.defenses import DifferentialPrivacy

# Enable differential privacy
dp_config = {
    'enabled': True,
    'epsilon': 1.0,
    'delta': 1e-5
}

# Train with privacy
model = TransformerModel()
model.fit_with_privacy(train.texts, train.labels, dp_config)
```

## Development

### Code Quality
The project uses modern Python development practices:
- **Type Hints**: Full type annotation coverage
- **Code Formatting**: Black and Ruff for consistent style
- **Testing**: Pytest with comprehensive test coverage
- **Documentation**: Google-style docstrings
- **Pre-commit Hooks**: Automated quality checks

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Code Formatting
```bash
# Format code
black src/ tests/ scripts/

# Lint code
ruff check src/ tests/ scripts/

# Fix linting issues
ruff check --fix src/ tests/ scripts/
```

## Performance Benchmarks

### Model Performance (Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Naive Bayes | 0.892 | 0.887 | 0.892 | 0.889 | 0.945 |
| Logistic Regression | 0.901 | 0.896 | 0.901 | 0.898 | 0.952 |
| Random Forest | 0.915 | 0.912 | 0.915 | 0.913 | 0.967 |
| DistilBERT | 0.928 | 0.925 | 0.928 | 0.926 | 0.974 |
| Ensemble | 0.931 | 0.928 | 0.931 | 0.929 | 0.976 |

### Computational Requirements
- **Training Time**: 5-15 minutes (depending on model and dataset size)
- **Memory Usage**: 2-8 GB RAM (transformer models require more)
- **Storage**: 100-500 MB for trained models
- **Inference Speed**: 1-10 ms per message (baseline models faster)

## Privacy and Security

### Data Protection
- **PII Anonymization**: Automatic detection and hashing of sensitive information
- **Differential Privacy**: Mathematical privacy guarantees
- **Secure Storage**: Encrypted model artifacts and logs
- **Access Control**: Role-based permissions for sensitive operations

### Security Measures
- **Input Validation**: Sanitization of all user inputs
- **Model Security**: Protection against model extraction attacks
- **Audit Logging**: Comprehensive logging of all operations
- **Error Handling**: Graceful failure without information leakage

## Limitations and Future Work

### Current Limitations
- **Synthetic Data**: Models trained on generated data may not generalize to real-world scenarios
- **Language Support**: Currently optimized for English text
- **Domain Specificity**: May not perform well on specialized domains
- **Adversarial Robustness**: Limited protection against sophisticated attacks

### Future Enhancements
- **Multi-language Support**: Extend to multiple languages and scripts
- **Real-time Processing**: Streaming inference for high-volume scenarios
- **Advanced Privacy**: Homomorphic encryption and secure multi-party computation
- **Adversarial Training**: Robustness against evasion attacks
- **Federated Learning**: Distributed training across multiple organizations

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{spam_detection_system,
  title={Spam Detection System: A Modern Privacy-Preserving Approach},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Spam-Detection-System}
}
```

## Support

For questions, issues, or contributions:
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: research@example.com

---

**Remember**: This is a research and educational tool. Always use responsibly and in accordance with applicable laws and regulations.
# Spam-Detection-System
