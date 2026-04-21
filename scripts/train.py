#!/usr/bin/env python3
"""Training script for spam detection system."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import create_synthetic_dataset, create_splits
from models import BaselineModel, TransformerModel, EnsembleModel
from eval import EvaluationMetrics, create_leaderboard
from utils import Config, setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train spam detection models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--data-size", type=int, default=1000, help="Size of synthetic dataset")
    parser.add_argument("--models", nargs="+", default=["baseline", "transformer", "ensemble"], 
                       help="Models to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    setup_logging(
        level=config.get("logging.level", "INFO"),
        log_file=config.get("logging.file")
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting spam detection model training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to train: {args.models}")
    
    # Create synthetic dataset
    logger.info(f"Creating synthetic dataset with {args.data_size} samples")
    dataset = create_synthetic_dataset(n_samples=args.data_size, seed=args.seed)
    
    # Create train/val/test splits
    train_dataset, val_dataset, test_dataset = create_splits(
        dataset,
        train_size=config.get("data.train_size", 0.8),
        val_size=config.get("data.val_size", 0.1),
        test_size=config.get("data.test_size", 0.1),
        seed=args.seed
    )
    
    logger.info(f"Dataset splits: train={len(train_dataset.texts)}, "
               f"val={len(val_dataset.texts)}, test={len(test_dataset.texts)}")
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics(threshold=config.get("evaluation.threshold", 0.5))
    
    # Train models
    model_results = {}
    
    for model_name in args.models:
        logger.info(f"Training {model_name} model")
        
        try:
            if model_name == "baseline":
                model = BaselineModel(
                    model_type=config.get("models.baseline.type", "naive_bayes"),
                    model_params=config.get("models.baseline", {})
                )
                
                # Extract features for baseline model
                from src.data.preprocessing import FeatureExtractor
                import numpy as np
                feature_extractor = FeatureExtractor()
                tfidf_features, rule_features = feature_extractor.fit_transform(train_dataset.texts)
                X_train = np.hstack([tfidf_features, rule_features])
                
                # Fit model
                model.fit(X_train, train_dataset.labels)
                
                # Evaluate on test set
                tfidf_test, rule_test = feature_extractor.transform(test_dataset.texts)
                X_test = np.hstack([tfidf_test, rule_test])
                
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Compute metrics
                metrics = evaluator.compute_metrics(test_dataset.labels, y_pred, y_proba)
                model_results[model_name] = metrics
                
                # Save model
                model.save(output_dir / f"{model_name}_model.pkl")
                
            elif model_name == "transformer":
                model = TransformerModel(
                    model_name=config.get("models.transformer.model_name", "distilbert-base-uncased"),
                    max_length=config.get("models.transformer.max_length", 512),
                    batch_size=config.get("models.transformer.batch_size", 16),
                    learning_rate=config.get("models.transformer.learning_rate", 2e-5),
                    num_epochs=config.get("models.transformer.num_epochs", 3),
                    device=config.get("training.device", "auto")
                )
                
                # Fit model
                model.fit(
                    train_dataset.texts,
                    train_dataset.labels,
                    val_dataset.texts,
                    val_dataset.labels
                )
                
                # Evaluate on test set
                y_pred = model.predict(test_dataset.texts)
                y_proba = model.predict_proba(test_dataset.texts)
                
                # Compute metrics
                metrics = evaluator.compute_metrics(test_dataset.labels, y_pred, y_proba)
                model_results[model_name] = metrics
                
                # Save model
                model.save(output_dir / f"{model_name}_model.pth")
                
            elif model_name == "ensemble":
                # Create ensemble with baseline and transformer
                baseline_model = BaselineModel()
                transformer_model = TransformerModel(
                    model_name=config.get("models.transformer.model_name", "distilbert-base-uncased"),
                    max_length=config.get("models.transformer.max_length", 512),
                    batch_size=config.get("models.transformer.batch_size", 16),
                    learning_rate=config.get("models.transformer.learning_rate", 2e-5),
                    num_epochs=config.get("models.transformer.num_epochs", 3),
                    device=config.get("training.device", "auto")
                )
                
                ensemble = EnsembleModel(
                    models=[baseline_model, transformer_model],
                    weights=config.get("models.ensemble.weights", [0.3, 0.7])
                )
                
                # Fit ensemble
                ensemble.fit(
                    train_dataset.texts,
                    train_dataset.labels,
                    val_dataset.texts,
                    val_dataset.labels
                )
                
                # Evaluate on test set
                y_pred = ensemble.predict(test_dataset.texts)
                y_proba = ensemble.predict_proba(test_dataset.texts)
                
                # Compute metrics
                metrics = evaluator.compute_metrics(test_dataset.labels, y_pred, y_proba)
                model_results[model_name] = metrics
                
                # Save model
                ensemble.save(output_dir / f"{model_name}_model.pkl")
            
            logger.info(f"{model_name} model training completed")
            logger.info(f"Test metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name} model: {e}")
            continue
    
    # Create leaderboard
    if model_results:
        leaderboard = create_leaderboard(model_results, sort_by='f1')
        logger.info("Model Leaderboard:")
        logger.info(f"\n{leaderboard.to_string()}")
        
        # Save leaderboard
        leaderboard.to_csv(output_dir / "leaderboard.csv", index=True)
        
        # Save results
        import json
        with open(output_dir / "results.json", "w") as f:
            json.dump(model_results, f, indent=2)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
