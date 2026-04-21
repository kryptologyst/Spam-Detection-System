"""Dataset classes and synthetic data generation."""

import hashlib
import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SpamDataset:
    """Dataset class for spam detection with privacy-preserving features."""

    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        metadata: Optional[Dict] = None,
        anonymize: bool = True
    ) -> None:
        """Initialize dataset.
        
        Args:
            texts: List of text messages
            labels: List of labels ('spam' or 'ham')
            metadata: Optional metadata dictionary
            anonymize: Whether to anonymize PII in texts
        """
        self.texts = texts
        self.labels = labels
        self.metadata = metadata or {}
        self.anonymize = anonymize
        
        if anonymize:
            self.texts = self._anonymize_texts(self.texts)
        
        self._validate_data()

    def _anonymize_texts(self, texts: List[str]) -> List[str]:
        """Anonymize PII in texts.
        
        Args:
            texts: List of texts to anonymize
            
        Returns:
            List of anonymized texts
        """
        anonymized = []
        
        for text in texts:
            # Hash email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                         lambda m: f"email_{hashlib.md5(m.group().encode()).hexdigest()[:8]}", text)
            
            # Hash phone numbers
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 
                         lambda m: f"phone_{hashlib.md5(m.group().encode()).hexdigest()[:8]}", text)
            
            # Hash IP addresses
            text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 
                         lambda m: f"ip_{hashlib.md5(m.group().encode()).hexdigest()[:8]}", text)
            
            anonymized.append(text)
        
        return anonymized

    def _validate_data(self) -> None:
        """Validate dataset integrity."""
        if len(self.texts) != len(self.labels):
            raise ValueError("Number of texts and labels must match")
        
        valid_labels = {'spam', 'ham'}
        invalid_labels = set(self.labels) - valid_labels
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}")
        
        logger.info(f"Dataset loaded: {len(self.texts)} samples, "
                   f"{sum(1 for l in self.labels if l == 'spam')} spam, "
                   f"{sum(1 for l in self.labels if l == 'ham')} ham")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.
        
        Returns:
            DataFrame with texts and labels
        """
        return pd.DataFrame({
            'text': self.texts,
            'label': self.labels
        })

    def get_class_weights(self) -> Dict[str, float]:
        """Calculate class weights for imbalanced learning.
        
        Returns:
            Dictionary with class weights
        """
        from collections import Counter
        counts = Counter(self.labels)
        total = len(self.labels)
        
        weights = {}
        for label, count in counts.items():
            weights[label] = total / (len(counts) * count)
        
        return weights


def create_synthetic_dataset(
    n_samples: int = 1000,
    spam_ratio: float = 0.3,
    seed: int = 42
) -> SpamDataset:
    """Create synthetic spam detection dataset.
    
    Args:
        n_samples: Total number of samples
        spam_ratio: Ratio of spam samples
        seed: Random seed
        
    Returns:
        SpamDataset instance
    """
    np.random.seed(seed)
    
    # Spam templates
    spam_templates = [
        "Congratulations! You've won a ${amount} {prize}. Click here to claim now!",
        "URGENT! Your {account} has been suspended. Update immediately!",
        "Limited time offer: {product} at {discount}% off. Act now!",
        "You have been selected for a ${amount} gift card! Don't miss out!",
        "Get rich quick with this {investment} opportunity!",
        "Free {product} - no strings attached! Click here!",
        "Your {service} subscription expires in 24 hours. Renew now!",
        "Exclusive offer: {product} for only ${price}. Limited stock!",
        "You've been chosen for a special {offer}. Claim your prize!",
        "Make money from home with this {method}. Start today!"
    ]
    
    # Ham templates
    ham_templates = [
        "Meeting scheduled for {time} today. Please be on time.",
        "Can you review the attached {document} before {deadline}?",
        "Let's catch up over {meal} next {day}.",
        "Please find the {document} attached for your review.",
        "Thanks for your help with the {project} yesterday.",
        "The {meeting} has been moved to {time} in {location}.",
        "Could you please send me the {information} by {deadline}?",
        "Looking forward to our {event} this {day}.",
        "Please confirm your attendance for the {meeting}.",
        "The {report} is ready for your review."
    ]
    
    # Template variables
    spam_vars = {
        'amount': ['100', '500', '1000', '5000', '10000'],
        'prize': ['Walmart gift card', 'Amazon voucher', 'cash prize', 'iPhone', 'laptop'],
        'account': ['bank account', 'PayPal account', 'credit card', 'email account'],
        'product': ['Viagra', 'weight loss pills', 'investment course', 'crypto guide'],
        'discount': ['50', '75', '90', '95'],
        'investment': ['crypto', 'forex', 'stocks', 'real estate'],
        'service': ['Netflix', 'Spotify', 'Prime', 'iCloud'],
        'price': ['9.99', '19.99', '29.99', '49.99'],
        'offer': ['discount', 'deal', 'promotion', 'special'],
        'method': ['system', 'program', 'strategy', 'technique']
    }
    
    ham_vars = {
        'time': ['3 PM', '2:30 PM', '4 PM', '10 AM', '11:30 AM'],
        'document': ['report', 'proposal', 'contract', 'invoice', 'presentation'],
        'deadline': ['tomorrow', 'Friday', 'next week', 'end of month'],
        'meal': ['lunch', 'dinner', 'coffee', 'breakfast'],
        'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'project': ['website', 'app', 'campaign', 'analysis'],
        'meeting': ['team meeting', 'client call', 'review session', 'planning session'],
        'location': ['conference room', 'office', 'cafeteria', 'library'],
        'information': ['details', 'specifications', 'requirements', 'data'],
        'event': ['conference', 'workshop', 'seminar', 'training']
    }
    
    def fill_template(template: str, variables: Dict[str, List[str]]) -> str:
        """Fill template with random variables."""
        result = template
        for var, values in variables.items():
            if f"{{{var}}}" in result:
                result = result.replace(f"{{{var}}}", np.random.choice(values))
        return result
    
    # Generate samples
    n_spam = int(n_samples * spam_ratio)
    n_ham = n_samples - n_spam
    
    texts = []
    labels = []
    
    # Generate spam samples
    for _ in range(n_spam):
        template = np.random.choice(spam_templates)
        text = fill_template(template, spam_vars)
        texts.append(text)
        labels.append('spam')
    
    # Generate ham samples
    for _ in range(n_ham):
        template = np.random.choice(ham_templates)
        text = fill_template(template, ham_vars)
        texts.append(text)
        labels.append('ham')
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return SpamDataset(texts, labels, anonymize=True)


def create_splits(
    dataset: SpamDataset,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify: bool = True,
    seed: int = 42
) -> Tuple[SpamDataset, SpamDataset, SpamDataset]:
    """Create train/validation/test splits.
    
    Args:
        dataset: Input dataset
        train_size: Training set size ratio
        val_size: Validation set size ratio
        test_size: Test set size ratio
        stratify: Whether to stratify by label
        seed: Random seed
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Split sizes must sum to 1.0")
    
    df = dataset.to_dataframe()
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df['label'] if stratify else None,
        random_state=seed
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df['label'] if stratify else None,
        random_state=seed
    )
    
    # Create dataset objects
    train_dataset = SpamDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        anonymize=False  # Already anonymized
    )
    
    val_dataset = SpamDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        anonymize=False
    )
    
    test_dataset = SpamDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        anonymize=False
    )
    
    return train_dataset, val_dataset, test_dataset
