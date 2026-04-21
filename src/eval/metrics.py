"""Evaluation metrics for spam detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for spam detection."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize evaluation metrics.
        
        Args:
            threshold: Classification threshold for binary predictions
        """
        self.threshold = threshold

    def compute_metrics(
        self,
        y_true: Union[List[str], np.ndarray],
        y_pred: Union[List[str], np.ndarray],
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Class-specific metrics
        for class_name in ['ham', 'spam']:
            if class_name in y_true:
                metrics[f'{class_name}_precision'] = precision_score(
                    y_true, y_pred, labels=[class_name], average='binary', pos_label=class_name
                )
                metrics[f'{class_name}_recall'] = recall_score(
                    y_true, y_pred, labels=[class_name], average='binary', pos_label=class_name
                )
                metrics[f'{class_name}_f1'] = f1_score(
                    y_true, y_pred, labels=[class_name], average='binary', pos_label=class_name
                )
        
        # Probability-based metrics
        if y_proba is not None:
            # Convert labels to binary for AUC calculation
            y_binary = (y_true == 'spam').astype(int)
            
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                # Multi-class probabilities
                spam_proba = y_proba[:, 1]
            else:
                # Binary probabilities
                spam_proba = y_proba
            
            metrics['auc'] = roc_auc_score(y_binary, spam_proba)
            metrics['aucpr'] = average_precision_score(y_binary, spam_proba)
            
            # Precision at different recall levels
            precision, recall, _ = precision_recall_curve(y_binary, spam_proba)
            metrics['precision_at_90_recall'] = self._precision_at_recall(precision, recall, 0.9)
            metrics['precision_at_95_recall'] = self._precision_at_recall(precision, recall, 0.95)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics

    def _precision_at_recall(self, precision: np.ndarray, recall: np.ndarray, target_recall: float) -> float:
        """Calculate precision at a specific recall level.
        
        Args:
            precision: Precision values
            recall: Recall values
            target_recall: Target recall level
            
        Returns:
            Precision at target recall
        """
        # Find the closest recall value
        idx = np.argmin(np.abs(recall - target_recall))
        return precision[idx]

    def compute_precision_at_k(
        self,
        y_true: Union[List[str], np.ndarray],
        y_proba: np.ndarray,
        k_values: List[int] = [10, 50, 100]
    ) -> Dict[str, float]:
        """Compute precision at K for ranking evaluation.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with precision@K metrics
        """
        y_true = np.array(y_true)
        
        # Convert to binary
        y_binary = (y_true == 'spam').astype(int)
        
        # Get spam probabilities
        if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
            spam_proba = y_proba[:, 1]
        else:
            spam_proba = y_proba
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(spam_proba)[::-1]
        sorted_labels = y_binary[sorted_indices]
        
        metrics = {}
        for k in k_values:
            if k <= len(sorted_labels):
                top_k_labels = sorted_labels[:k]
                precision_at_k = np.mean(top_k_labels)
                metrics[f'precision_at_{k}'] = precision_at_k
        
        return metrics

    def compute_operational_metrics(
        self,
        y_true: Union[List[str], np.ndarray],
        y_proba: np.ndarray,
        thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> pd.DataFrame:
        """Compute operational metrics at different thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            DataFrame with operational metrics
        """
        y_true = np.array(y_true)
        y_binary = (y_true == 'spam').astype(int)
        
        # Get spam probabilities
        if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
            spam_proba = y_proba[:, 1]
        else:
            spam_proba = y_proba
        
        results = []
        for threshold in thresholds:
            y_pred = (spam_proba >= threshold).astype(int)
            y_pred_labels = ['spam' if p else 'ham' for p in y_pred]
            
            metrics = self.compute_metrics(y_true, y_pred_labels, y_proba)
            metrics['threshold'] = threshold
            metrics['alert_volume'] = np.sum(y_pred)
            metrics['alert_rate'] = np.mean(y_pred)
            
            results.append(metrics)
        
        return pd.DataFrame(results)


def create_leaderboard(
    model_results: Dict[str, Dict[str, float]],
    sort_by: str = 'f1'
) -> pd.DataFrame:
    """Create a leaderboard from model evaluation results.
    
    Args:
        model_results: Dictionary mapping model names to metrics
        sort_by: Metric to sort by
        
    Returns:
        DataFrame with leaderboard
    """
    # Convert to DataFrame
    df = pd.DataFrame(model_results).T
    
    # Sort by specified metric
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    priority_columns = ['rank', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'aucpr']
    other_columns = [col for col in df.columns if col not in priority_columns]
    df = df[priority_columns + other_columns]
    
    return df


def compare_models(
    model_results: Dict[str, Dict[str, float]],
    baseline_model: str = None
) -> pd.DataFrame:
    """Compare models against a baseline.
    
    Args:
        model_results: Dictionary mapping model names to metrics
        baseline_model: Name of baseline model (if None, use first model)
        
    Returns:
        DataFrame with comparison results
    """
    if baseline_model is None:
        baseline_model = list(model_results.keys())[0]
    
    if baseline_model not in model_results:
        raise ValueError(f"Baseline model '{baseline_model}' not found in results")
    
    baseline_metrics = model_results[baseline_model]
    
    comparison = []
    for model_name, metrics in model_results.items():
        if model_name == baseline_model:
            continue
        
        comparison_row = {'model': model_name}
        for metric, value in metrics.items():
            baseline_value = baseline_metrics.get(metric, 0)
            if baseline_value != 0:
                improvement = ((value - baseline_value) / baseline_value) * 100
                comparison_row[f'{metric}_improvement'] = improvement
            else:
                comparison_row[f'{metric}_improvement'] = 0
        
        comparison.append(comparison_row)
    
    return pd.DataFrame(comparison)
