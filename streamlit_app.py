#!/usr/bin/env python3
"""
Streamlit Fake News Detection App
Combines training and prediction in a user-friendly web interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import os
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fake-news {
        color: #d62728;
        font-weight: bold;
    }
    .real-news {
        color: #2ca02c;
        font-weight: bold;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    """Clean text using the same approach as training"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_training_script():
    """Run the external training script and stream output."""
    st.info("Starting training using train_simple_model.py ... This may take several minutes.")
    with st.spinner('Training in progress...'):
        try:
            process = subprocess.run(
                [sys.executable, 'train_simple_model.py'],
                capture_output=True,
                text=True,
                check=False
            )
            if process.stdout:
                st.text(process.stdout)
            if process.stderr:
                st.text(process.stderr)
            if process.returncode == 0:
                st.success("Training completed and artifacts saved in ./artifacts")
            else:
                st.error(f"Training script exited with code {process.returncode}")
        except Exception as e:
            st.error(f"Failed to run training script: {e}")

def predict_text_using_script(text):
    """Call the external predict_fake_news.py script to make predictions"""
    try:
        # Call the external script with the text
        process = subprocess.run(
            [
                sys.executable, 'predict_fake_news.py',
                '--model', './artifacts/best_model.joblib',
                '--vectorizer', './artifacts/tfidf_vectorizer.joblib',
                '--text', text
            ],
            capture_output=True,
            text=True,
            check=False
        )
        
        if process.returncode != 0:
            st.error(f"Prediction script failed with return code {process.returncode}")
            if process.stderr:
                st.error(f"Error: {process.stderr}")
            return None, None, None
        
        # Parse the output to extract prediction and decision score
        output = process.stdout
        
        # Look for the prediction label (FAKE or REAL)
        prediction_label = None
        decision_score = None
        processed_text = ""
        
        lines = output.split('\n')
        for i, line in enumerate(lines):
            # Look for "Example 1: FAKE" or "Example 1: REAL"
            if line.startswith("Example 1:"):
                prediction_label = line.split(": ")[1].strip()
            
            # Look for "Decision score: X.XXXX"
            if line.startswith("Decision score:"):
                try:
                    decision_score = float(line.split(": ")[1])
                except (IndexError, ValueError):
                    pass
            
            # Look for processed text (after "Processed text excerpt:")
            if line.startswith("Processed text excerpt:") and i + 1 < len(lines):
                processed_text = lines[i + 1].strip()
        
        # Convert label to numeric prediction
        prediction = 0 if prediction_label == "FAKE" else 1 if prediction_label == "REAL" else None
        
        if prediction is None or decision_score is None:
            st.error("Could not parse prediction results from script output")
            return None, None, None
        
        return prediction, decision_score, processed_text
        
    except Exception as e:
        st.error(f"Error calling prediction script: {str(e)}")
        return None, None, None


def predict_multiple_texts_using_script(texts):
    """Call the external predict_fake_news.py script for multiple texts"""
    try:
        # Create a temporary file with the texts
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            for text in texts:
                temp_file.write(text + '\n')
            temp_file_path = temp_file.name
        
        try:
            # Call the external script with the file
            process = subprocess.run(
                [
                    sys.executable, 'predict_fake_news.py',
                    '--model', './artifacts/best_model.joblib',
                    '--vectorizer', './artifacts/tfidf_vectorizer.joblib',
                    '--file', temp_file_path
                ],
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                st.error(f"Prediction script failed with return code {process.returncode}")
                if process.stderr:
                    st.error(f"Error: {process.stderr}")
                return []
            
            # Parse the output to extract predictions
            output = process.stdout
            results = []
            
            lines = output.split('\n')
            current_example = {}
            
            for i, line in enumerate(lines):
                # Look for "Example N: FAKE" or "Example N: REAL"
                if line.startswith("Example ") and ": " in line:
                    if current_example:  # Save previous example
                        results.append(current_example)
                    
                    example_num = int(line.split()[1].rstrip(':'))
                    prediction_label = line.split(": ")[1].strip()
                    prediction = 0 if prediction_label == "FAKE" else 1
                    
                    current_example = {
                        'text': texts[example_num - 1] if example_num <= len(texts) else "",
                        'prediction': prediction,
                        'decision_score': None,
                        'cleaned_text': ""
                    }
                
                # Look for "Decision score: X.XXXX"
                elif line.startswith("Decision score:") and current_example:
                    try:
                        current_example['decision_score'] = float(line.split(": ")[1])
                    except (IndexError, ValueError):
                        pass
                
                # Look for processed text
                elif line.startswith("Processed text excerpt:") and i + 1 < len(lines):
                    if current_example:
                        current_example['cleaned_text'] = lines[i + 1].strip()
            
            # Don't forget the last example
            if current_example:
                results.append(current_example)
            
            return results
            
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        st.error(f"Error calling prediction script for multiple texts: {str(e)}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📰 Predict News", "📊 Model Info"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📰 Predict News":
        show_prediction_page()
    elif page == "📊 Model Info":
        show_model_info_page()

def show_home_page():
    """Display the home page"""
    st.markdown("""
    ## Welcome to the Fake News Detector! 🎯
    
    This application helps you:
    - **Train** a machine learning model to detect fake news
    - **Predict** whether news articles are fake or real
    - **Analyze** model performance and predictions
    
    ### How it works:
    1. **Train Model**: Click the button below to run training with `Fake.csv` and `True.csv`
    2. **Predict News**: Input any news text and get instant fake/real classification
    3. **Model Info**: View training statistics and model details
    
    ### Features:
    - 🤖 **Machine Learning**: Uses TF-IDF vectorization and LinearSVC
    - 📊 **One-click Training**: Train your model from local CSVs
    - 🔍 **Instant Prediction**: Get results with confidence scores
    - 📈 **Performance Metrics**: View accuracy and detailed reports
    """)
    
    # Training controls
    st.subheader("🛠 Training")
    if st.button("🚀 Run Training (train_simple_model.py)"):
        if os.path.exists('dataset/Fake.csv') and os.path.exists('dataset/True.csv'):
            run_training_script()
        else:
            st.error("Fake.csv and/or True.csv not found in the project directory.")

    # Check if model exists
    if os.path.exists('./artifacts/best_model.joblib') and os.path.exists('./artifacts/tfidf_vectorizer.joblib'):
        st.success("✅ **Model Ready**: You have a trained model available for predictions!")
        
        # Quick prediction demo
        st.subheader("🚀 Quick Demo")
        demo_text = st.text_area(
            "Try a quick prediction:",
            value="Enter some news text here to test the model...",
            height=100
        )
        
        if st.button("🔍 Predict", key="demo_predict"):
            if demo_text and demo_text != "Enter some news text here to test the model...":
                try:
                    model = joblib.load('./artifacts/best_model.joblib')
                    vectorizer = joblib.load('./artifacts/tfidf_vectorizer.joblib')
                    
                    # Use the external prediction script
                    prediction, decision_score, cleaned = predict_text_using_script(demo_text)
                    if prediction is None:
                        return
                    
                    label = "FAKE" if prediction == 0 else "REAL"
                    color_class = "fake-news" if label == "FAKE" else "real-news"
                    
                    st.markdown(f"<h3 class='{color_class}'>{label}</h3>", unsafe_allow_html=True)
                    st.metric("Confidence Score", f"{abs(decision_score):.4f}")
                    
                    # Confidence bar
                    confidence = min(abs(decision_score) / 2, 1.0)  # Normalize to 0-1
                    st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter some text to test.")
    else:
        st.info("ℹ️ **No Model Found**: Please go to the 'Train Model' page to create your first model.")

    # End of training controls

def show_prediction_page():
    """Display the prediction page"""
    st.header("📰 Predict News")
    
    # Check if model exists
    if not (os.path.exists('./artifacts/best_model.joblib') and os.path.exists('./artifacts/tfidf_vectorizer.joblib')):
        st.error("❌ **No trained model found!** Please train a model first.")
        st.info("Go to the 'Train Model' page to create your first model.")
        return
    
    try:
        # Load model
        model = joblib.load('./artifacts/best_model.joblib')
        vectorizer = joblib.load('./artifacts/tfidf_vectorizer.joblib')
        
        st.success("✅ Model loaded successfully!")
        
        # Input methods
        st.subheader("📝 Input Method")
        input_method = st.radio(
            "Choose input method:",
            ["Single Text", "Multiple Texts", "File Upload"],
            horizontal=True
        )
        
        if input_method == "Single Text":
            st.subheader("🔍 Single Text Prediction")
            text_input = st.text_area(
                "Enter news text:",
                height=150,
                placeholder="Paste your news article or headline here..."
            )
            
            if st.button("🔍 Predict", type="primary"):
                if text_input.strip():
                    prediction, decision_score, cleaned_text = predict_text_using_script(text_input)
                    
                    if prediction is not None:
                        display_prediction_result(text_input, prediction, decision_score, cleaned_text)
                else:
                    st.warning("Please enter some text to predict.")
        
        elif input_method == "Multiple Texts":
            st.subheader("📰 Multiple Texts Prediction")
            texts_input = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="Text 1\nText 2\nText 3\n..."
            )
            
            if st.button("🔍 Predict All", type="primary"):
                if texts_input.strip():
                    texts = [text.strip() for text in texts_input.split('\n') if text.strip()]
                    
                    # Use the external script for multiple predictions
                    results = predict_multiple_texts_using_script(texts)
                    
                    if results:
                        display_multiple_predictions(results)
                else:
                    st.warning("Please enter some texts to predict.")
        
        else:  # File Upload
            st.subheader("📁 File Upload Prediction")
            uploaded_file = st.file_uploader(
                "Upload a text file (one article per line):",
                type=['txt', 'csv'],
                key="prediction_file"
            )
            
            if uploaded_file and st.button("🔍 Predict from File", type="primary"):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            texts = df['text'].tolist()
                        else:
                            st.error("CSV must contain a 'text' column!")
                            return
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        texts = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    # Use the external script for file upload predictions
                    with st.spinner('Processing file...'):
                        results = predict_multiple_texts_using_script(texts)
                    
                    if results:
                        display_multiple_predictions(results)
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

def display_prediction_result(original_text, prediction, decision_score, cleaned_text):
    """Display a single prediction result"""
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    # Result display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        label = "FAKE" if prediction == 0 else "REAL"
        color_class = "fake-news" if label == "FAKE" else "real-news"
        
        st.markdown(f"<h2 class='{color_class}'>{label}</h2>", unsafe_allow_html=True)
        
        # Confidence visualization
        confidence = min(abs(decision_score) / 2, 1.0)  # Normalize to 0-1
        st.metric("Confidence Score", f"{abs(decision_score):.4f}")
        st.progress(confidence)
        
        # Decision score interpretation
        if decision_score > 1.0:
            confidence_level = "Very High"
        elif decision_score > 0.5:
            confidence_level = "High"
        elif decision_score > 0.1:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        st.info(f"Confidence Level: **{confidence_level}**")
    
    with col2:
        st.markdown("**Decision Score:**")
        if decision_score > 0:
            st.success(f"+{decision_score:.4f} (REAL)")
        else:
            st.error(f"{decision_score:.4f} (FAKE)")
    
    # Text analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📰 Original Text:**")
        st.text_area("Original Text", value=original_text, height=100, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.markdown("**🧹 Cleaned Text:**")
        st.text_area("Cleaned Text", value=cleaned_text, height=100, disabled=True, label_visibility="collapsed")

def display_multiple_predictions(results):
    """Display multiple prediction results"""
    st.markdown("---")
    st.subheader("📊 Multiple Predictions Results")
    
    # Summary statistics
    total = len(results)
    fake_count = sum(1 for r in results if r['prediction'] == 0)
    real_count = sum(1 for r in results if r['prediction'] == 1)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Texts", total)
    with col2:
        st.metric("Fake", fake_count)
    with col3:
        st.metric("Real", real_count)
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    # Prepare data for display
    display_data = []
    for i, result in enumerate(results):
        display_data.append({
            'Text #': i + 1,
            'Prediction': 'FAKE' if result['prediction'] == 0 else 'REAL',
            'Confidence': f"{abs(result['decision_score']):.4f}",
            'Decision Score': f"{result['decision_score']:.4f}",
            'Text Preview': result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
        })
    
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display, use_container_width=True)
    
    # Download results
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name=f"fake_news_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_model_info_page():
    """Display model information page"""
    st.header("📊 Model Information")
    
    # Check if model exists
    if not (os.path.exists('./artifacts/best_model.joblib') and os.path.exists('./artifacts/tfidf_vectorizer.joblib')):
        st.error("❌ **No trained model found!** Please train a model first.")
        st.info("Go to the 'Train Model' page to create your first model.")
        return
    
    try:
        # Load model
        model = joblib.load('./artifacts/best_model.joblib')
        vectorizer = joblib.load('./artifacts/tfidf_vectorizer.joblib')
        
        st.success("✅ Model loaded successfully!")
        
        # Model details
        st.subheader("🤖 Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Type:**")
            st.info(f"{type(model).__name__}")
            
            st.markdown("**Vectorizer Type:**")
            st.info(f"{type(vectorizer).__name__}")
            
            st.markdown("**Vocabulary Size:**")
            st.info(f"{len(vectorizer.get_feature_names_out()):,} features")
        
        with col2:
            st.markdown("**Model Parameters:**")
            params = model.get_params()
            for key, value in params.items():
                st.text(f"{key}: {value}")
        
        # Training results if available
        if 'training_results' in st.session_state:
            st.markdown("---")
            st.subheader("📈 Training Results")
            
            results = st.session_state['training_results']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
            
            with col2:
                st.metric("Training Samples", f"{results['train_size']:,}")
            
            with col3:
                st.metric("Test Samples", f"{results['test_size']:,}")
            
            # Feature importance (top features)
            st.markdown("---")
            st.subheader("🔍 Feature Analysis")
            
            # Show some sample features
            feature_names = vectorizer.get_feature_names_out()
            st.info(f"Total features: {len(feature_names):,}")
            
            # Sample features
            st.markdown("**Sample Features:**")
            sample_features = feature_names[:20]
            st.text(", ".join(sample_features))
        
        # Model file info
        st.markdown("---")
        st.subheader("💾 Model Files")
        
        model_size = os.path.getsize('./artifacts/best_model.joblib') / (1024 * 1024)  # MB
        vectorizer_size = os.path.getsize('./artifacts/tfidf_vectorizer.joblib') / (1024 * 1024)  # MB
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model File Size", f"{model_size:.2f} MB")
        
        with col2:
            st.metric("Vectorizer File Size", f"{vectorizer_size:.2f} MB")
        
        # File timestamps
        model_time = datetime.fromtimestamp(os.path.getmtime('./artifacts/best_model.joblib'))
        vectorizer_time = datetime.fromtimestamp(os.path.getmtime('./artifacts/tfidf_vectorizer.joblib'))
        
        st.markdown("**Last Modified:**")
        st.text(f"Model: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.text(f"Vectorizer: {vectorizer_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()
