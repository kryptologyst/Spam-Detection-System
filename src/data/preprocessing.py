"""Text preprocessing and feature extraction utilities."""

import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing pipeline for spam detection."""

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        min_length: int = 2
    ) -> None:
        """Initialize text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase
            min_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.min_length = min_length
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
                self.stop_words = set()
                self.lemmatizer = None
        else:
            logger.warning("NLTK not available, using basic preprocessing")
            self.stop_words = set()
            self.lemmatizer = None

    def preprocess(self, text: str) -> str:
        """Preprocess a single text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize and process words
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()
        else:
            tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            # Filter by length
            if len(token) < self.min_length:
                continue
            
            # Remove stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Lemmatize
            if self.lemmatize and self.lemmatizer:
                try:
                    token = self.lemmatizer.lemmatize(token)
                except Exception:
                    pass  # Keep original token if lemmatization fails
            
            processed_tokens.append(token)
        
        return ' '.join(processed_tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class RuleBasedFeatures:
    """Rule-based feature extraction for spam detection."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialize rule-based feature extractor.
        
        Args:
            config: Configuration dictionary with patterns and keywords
        """
        self.config = config or {}
        
        # Default patterns
        self.spam_keywords = self.config.get('spam_keywords', [
            'free', 'urgent', 'click', 'winner', 'congratulations', 
            'limited time', 'act now', 'guaranteed', 'no risk', 'money back'
        ])
        
        self.suspicious_patterns = self.config.get('suspicious_patterns', [
            r'\$\d+',  # Money amounts
            r'\b\d{4,}\b',  # Long numbers
            r'!!!+',  # Multiple exclamation marks
            r'\b[A-Z]{3,}\b'  # All caps words
        ])
        
        self.url_pattern = self.config.get('url_pattern', r'https?://[^\s]+')
        self.phone_pattern = self.config.get('phone_pattern', r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract rule-based features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Text length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Spam keyword features
        text_lower = text.lower()
        spam_keyword_count = sum(1 for keyword in self.spam_keywords if keyword in text_lower)
        features['spam_keyword_ratio'] = spam_keyword_count / max(len(text.split()), 1)
        features['has_spam_keywords'] = float(spam_keyword_count > 0)
        
        # Suspicious pattern features
        suspicious_count = 0
        for pattern in self.suspicious_patterns:
            suspicious_count += len(re.findall(pattern, text))
        features['suspicious_pattern_ratio'] = suspicious_count / max(len(text.split()), 1)
        features['has_suspicious_patterns'] = float(suspicious_count > 0)
        
        # URL features
        url_count = len(re.findall(self.url_pattern, text))
        features['url_count'] = url_count
        features['has_urls'] = float(url_count > 0)
        
        # Phone number features
        phone_count = len(re.findall(self.phone_pattern, text))
        features['phone_count'] = phone_count
        features['has_phones'] = float(phone_count > 0)
        
        # Capitalization features
        upper_count = sum(1 for c in text if c.isupper())
        features['upper_ratio'] = upper_count / max(len(text), 1)
        features['has_excessive_caps'] = float(upper_count > len(text) * 0.3)
        
        # Punctuation features
        punct_count = sum(1 for c in text if c in '!?')
        features['exclamation_ratio'] = punct_count / max(len(text.split()), 1)
        features['has_excessive_punctuation'] = float(punct_count > 3)
        
        # Number features
        number_count = len(re.findall(r'\d+', text))
        features['number_ratio'] = number_count / max(len(text.split()), 1)
        features['has_numbers'] = float(number_count > 0)
        
        return features

    def extract_batch_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract features for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            DataFrame with extracted features
        """
        features_list = [self.extract_features(text) for text in texts]
        return pd.DataFrame(features_list)


class FeatureExtractor:
    """Combined feature extraction pipeline."""

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        rule_extractor: Optional[RuleBasedFeatures] = None,
        tfidf_config: Optional[Dict] = None
    ) -> None:
        """Initialize feature extractor.
        
        Args:
            preprocessor: Text preprocessor
            rule_extractor: Rule-based feature extractor
            tfidf_config: TF-IDF vectorizer configuration
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        self.rule_extractor = rule_extractor or RuleBasedFeatures()
        
        # TF-IDF configuration
        tfidf_config = tfidf_config or {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 10000),
            ngram_range=tfidf_config.get('ngram_range', (1, 2)),
            min_df=tfidf_config.get('min_df', 2),
            max_df=tfidf_config.get('max_df', 0.95),
            stop_words='english'
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        """Fit the feature extractor on training data.
        
        Args:
            texts: Training texts
            
        Returns:
            Self for chaining
        """
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(processed_texts)
        
        # Fit scaler on rule-based features
        rule_features = self.rule_extractor.extract_batch_features(texts)
        self.scaler.fit(rule_features)
        
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform texts to feature matrices.
        
        Args:
            texts: Input texts
            
        Returns:
            Tuple of (tfidf_features, rule_features)
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts).toarray()
        
        # Extract rule-based features
        rule_features = self.rule_extractor.extract_batch_features(texts)
        rule_features_scaled = self.scaler.transform(rule_features)
        
        return tfidf_features, rule_features_scaled

    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform texts.
        
        Args:
            texts: Input texts
            
        Returns:
            Tuple of (tfidf_features, rule_features)
        """
        return self.fit(texts).transform(texts)
