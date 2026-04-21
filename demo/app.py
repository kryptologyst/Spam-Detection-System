"""Streamlit demo application for spam detection system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import create_synthetic_dataset, SpamDataset
from models import BaselineModel, TransformerModel, EnsembleModel
from eval import EvaluationMetrics, ModelExplainer
from utils import Config, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .spam-prediction {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .ham-prediction {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'results' not in st.session_state:
    st.session_state.results = {}

@st.cache_data
def load_config():
    """Load configuration."""
    try:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        return Config(str(config_path))
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return None

@st.cache_data
def create_demo_dataset(n_samples: int = 500):
    """Create demo dataset."""
    set_seed(42)
    return create_synthetic_dataset(n_samples=n_samples, seed=42)

def load_model(model_path: str, model_type: str):
    """Load trained model."""
    try:
        if model_type == "baseline":
            return BaselineModel.load(model_path)
        elif model_type == "transformer":
            return TransformerModel.load(model_path)
        elif model_type == "ensemble":
            return EnsembleModel.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        st.error(f"Failed to load {model_type} model: {e}")
        return None

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🛡️ Spam Detection System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Disclaimer:</strong> This is a research and educational demonstration of spam detection techniques. 
    The system is designed for defensive purposes only and should not be used for production security operations 
    or exploitation. Results may be inaccurate and should not be relied upon for critical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    if config is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    available_models = ["baseline", "transformer", "ensemble"]
    selected_models = st.sidebar.multiselect(
        "Select models to use:",
        available_models,
        default=["baseline", "transformer"]
    )
    
    # Dataset size
    st.sidebar.subheader("Dataset Settings")
    dataset_size = st.sidebar.slider("Dataset size:", 100, 1000, 500)
    
    # Create dataset
    if st.sidebar.button("Generate Dataset") or st.session_state.dataset is None:
        with st.spinner("Generating synthetic dataset..."):
            st.session_state.dataset = create_demo_dataset(dataset_size)
        st.success(f"Generated dataset with {len(st.session_state.dataset.texts)} samples")
    
    # Load models
    st.sidebar.subheader("Model Loading")
    models_dir = Path(__file__).parent.parent / "models"
    
    for model_name in selected_models:
        if model_name not in st.session_state.models:
            model_path = models_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                with st.spinner(f"Loading {model_name} model..."):
                    model = load_model(str(model_path), model_name)
                    if model is not None:
                        st.session_state.models[model_name] = model
                        st.success(f"Loaded {model_name} model")
            else:
                st.warning(f"{model_name} model not found. Please train models first.")
    
    # Main content
    if st.session_state.dataset is None:
        st.info("Please generate a dataset first using the sidebar.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📊 Analyze", "📈 Evaluate", "🔧 Explain"])
    
    with tab1:
        st.header("Spam Detection")
        
        # Input text
        st.subheader("Enter Text to Analyze")
        input_text = st.text_area(
            "Message:",
            placeholder="Enter a message to check for spam...",
            height=100
        )
        
        # Example texts
        st.subheader("Example Texts")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Spam Example"):
                st.session_state.example_text = "Congratulations! You've won a $1000 gift card. Click here to claim now!"
        
        with col2:
            if st.button("Ham Example"):
                st.session_state.example_text = "Meeting scheduled for 3 PM today. Please be on time."
        
        if 'example_text' in st.session_state:
            input_text = st.session_state.example_text
            st.text_area("Message:", value=input_text, height=100, key="example_display")
        
        # Predict button
        if st.button("🔍 Predict", type="primary") and input_text.strip():
            if not st.session_state.models:
                st.error("No models loaded. Please load models first.")
                return
            
            # Make predictions
            predictions = {}
            probabilities = {}
            
            for model_name, model in st.session_state.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([input_text])[0]
                        pred = model.predict([input_text])[0]
                    else:
                        pred = model.predict([input_text])[0]
                        proba = [0.5, 0.5]  # Default probabilities
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = proba
                    
                except Exception as e:
                    st.error(f"Prediction failed for {model_name}: {e}")
                    continue
            
            # Display results
            if predictions:
                st.subheader("Prediction Results")
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Prediction': list(predictions.values()),
                    'Spam Probability': [prob[1] if len(prob) > 1 else prob[0] for prob in probabilities.values()],
                    'Ham Probability': [prob[0] if len(prob) > 1 else 1-prob[0] for prob in probabilities.values()]
                })
                
                # Display table
                st.dataframe(results_df, use_container_width=True)
                
                # Visualize probabilities
                fig = px.bar(
                    results_df,
                    x='Model',
                    y=['Ham Probability', 'Spam Probability'],
                    title="Prediction Probabilities by Model",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Consensus prediction
                spam_votes = sum(1 for pred in predictions.values() if pred == 'spam')
                total_votes = len(predictions)
                consensus = 'spam' if spam_votes > total_votes / 2 else 'ham'
                
                if consensus == 'spam':
                    st.markdown(f"""
                    <div class="spam-prediction">
                    <h3>🚨 SPAM DETECTED</h3>
                    <p>Consensus: {spam_votes}/{total_votes} models classified this as spam</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ham-prediction">
                    <h3>✅ LEGITIMATE MESSAGE</h3>
                    <p>Consensus: {total_votes - spam_votes}/{total_votes} models classified this as legitimate</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Dataset Analysis")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        dataset = st.session_state.dataset
        df = dataset.to_dataframe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Spam Samples", len(df[df['label'] == 'spam']))
        with col3:
            st.metric("Ham Samples", len(df[df['label'] == 'ham']))
        with col4:
            spam_ratio = len(df[df['label'] == 'spam']) / len(df)
            st.metric("Spam Ratio", f"{spam_ratio:.1%}")
        
        # Label distribution
        st.subheader("Label Distribution")
        label_counts = df['label'].value_counts()
        fig = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="Distribution of Spam vs Ham"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Text length analysis
        st.subheader("Text Length Analysis")
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df,
                x='text_length',
                color='label',
                title="Character Length Distribution",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df,
                x='word_count',
                color='label',
                title="Word Count Distribution",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab3:
        st.header("Model Evaluation")
        
        if not st.session_state.models:
            st.info("No models loaded. Please load models first.")
            return
        
        # Evaluation metrics
        st.subheader("Model Performance")
        
        # Create evaluation dataset
        eval_dataset = create_demo_dataset(200)
        
        # Evaluate models
        evaluator = EvaluationMetrics()
        results = {}
        
        for model_name, model in st.session_state.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict(eval_dataset.texts)
                    y_proba = model.predict_proba(eval_dataset.texts)
                else:
                    y_pred = model.predict(eval_dataset.texts)
                    y_proba = None
                
                metrics = evaluator.compute_metrics(eval_dataset.labels, y_pred, y_proba)
                results[model_name] = metrics
                
            except Exception as e:
                st.error(f"Evaluation failed for {model_name}: {e}")
                continue
        
        if results:
            # Create results DataFrame
            results_df = pd.DataFrame(results).T
            results_df = results_df.round(4)
            
            # Display metrics
            st.dataframe(results_df, use_container_width=True)
            
            # Visualize metrics
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
            
            if available_metrics:
                fig = px.bar(
                    results_df[available_metrics],
                    title="Model Performance Comparison",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            model_for_cm = st.selectbox("Select model for confusion matrix:", list(results.keys()))
            
            if model_for_cm in st.session_state.models:
                model = st.session_state.models[model_for_cm]
                try:
                    y_pred = model.predict(eval_dataset.texts)
                    
                    # Create confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(eval_dataset.labels, y_pred, labels=['ham', 'spam'])
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title=f"Confusion Matrix - {model_for_cm}",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['ham', 'spam'],
                        y=['ham', 'spam']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Failed to create confusion matrix: {e}")
    
    with tab4:
        st.header("Model Explainability")
        
        if not st.session_state.models:
            st.info("No models loaded. Please load models first.")
            return
        
        # Explanation input
        st.subheader("Explain Prediction")
        explain_text = st.text_area(
            "Text to explain:",
            placeholder="Enter text to get feature importance explanation...",
            height=100
        )
        
        model_for_explanation = st.selectbox(
            "Select model for explanation:",
            list(st.session_state.models.keys())
        )
        
        if st.button("🔍 Explain") and explain_text.strip():
            if model_for_explanation in st.session_state.models:
                model = st.session_state.models[model_for_explanation]
                
                try:
                    # Create explainer
                    explainer = ModelExplainer(model)
                    
                    # Get explanation
                    explanation = explainer.explain_prediction(explain_text, method="rule_based")
                    
                    # Display prediction
                    pred = explanation['prediction']
                    confidence = explanation['confidence']
                    
                    if pred == 'spam':
                        st.markdown(f"""
                        <div class="spam-prediction">
                        <h3>🚨 Prediction: SPAM</h3>
                        <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ham-prediction">
                        <h3>✅ Prediction: HAM</h3>
                        <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display feature importance
                    if 'feature_importance' in explanation:
                        st.subheader("Feature Importance")
                        features = explanation['feature_importance']
                        
                        if features:
                            # Create DataFrame
                            feature_df = pd.DataFrame(features, columns=['Feature', 'Importance'])
                            feature_df['Abs_Importance'] = abs(feature_df['Importance'])
                            feature_df = feature_df.sort_values('Abs_Importance', ascending=True)
                            
                            # Create horizontal bar chart
                            fig = px.bar(
                                feature_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Feature Importance",
                                color='Importance',
                                color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display table
                            st.dataframe(feature_df[['Feature', 'Importance']], use_container_width=True)
                        else:
                            st.info("No feature importance available for this model.")
                    
                except Exception as e:
                    st.error(f"Explanation failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
    Spam Detection System - Research & Educational Demo<br>
    Built with Streamlit, PyTorch, and scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
