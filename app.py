import streamlit as st
import pickle
import zipfile
import os

# --- Extract model from zip if not already extracted ---
def extract_model_zip(zip_path, extract_to=".", model_filename="sentiment_analysis_model.pkl"):
    if not os.path.exists(os.path.join(extract_to, model_filename)):
        if not os.path.exists(zip_path):
            st.error(f"Model zip file not found: {zip_path}")
            st.stop()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

# --- Load the model ---
@st.cache_resource
def load_model():
    zip_path = "sentiment_analysis_model.pkl.zip"  # File must be in same folder
    model_filename = "sentiment_analysis_model.pkl"
    extract_model_zip(zip_path, ".", model_filename)
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    return model

# --- Predict Function ---
def predict_sentiments(text_list, model):
    return model.predict(text_list)

# --- Load local files (HTML/CSS) ---
def load_local_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

# --- Load HTML/CSS Templates ---
html_template = load_local_file("index.html")
css_style = load_local_file("style.css")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
if css_style:
    st.markdown(f"<style>{css_style}</style>", unsafe_allow_html=True)
if html_template:
    st.markdown(html_template, unsafe_allow_html=True)
else:
    st.markdown("## 💬 Simple Sentiment Analyzer\nMade with ❤ using Streamlit")

# --- Load Model ---
model = load_model()

# --- Input Text Box ---
user_input_texts = st.text_area("📝 Enter multiple texts (one per line):", height=200, placeholder="Type here...")

# --- Predict Button ---
if st.button("🔍 Predict Sentiments"):
    lines = [line.strip() for line in user_input_texts.split("\n") if line.strip()]
    
    if not lines:
        st.warning("⚠ Please enter at least one non-empty line.")
    else:
        predictions = predict_sentiments(lines, model)
        sentiment_labels = {0: "Negative", 2: "Neutral", 4: "Positive"}
        st.subheader("📊 Prediction Results")

        for i, (text, label) in enumerate(zip(lines, predictions), 1):
            sentiment = sentiment_labels.get(label, "Unknown")
            color = {
                "Positive": "#4CAF50",
                "Neutral": "#FF9800",
                "Negative": "#F44336"
            }.get(sentiment, "#000000")

            st.markdown(f"""
                <div class="result-box" style="border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px;">
                    <strong>{i}. {text}</strong><br>
                    Sentiment: <span style="color: {color}; font-weight: bold;">{sentiment}</span> ({label})
                </div>
            """, unsafe_allow_html=True)
