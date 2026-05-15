# app.py Steam Review ABSA System

import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
import spacy
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Steam ABSA",
    page_icon="🎮",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #ffffff; }
    .title { font-size: 36px; font-weight: bold; color: #58a6ff; text-align: center; padding: 20px 0 5px 0; }
    .subtitle { font-size: 15px; color: #8b949e; text-align: center; margin-bottom: 25px; }
    .section { font-size: 16px; font-weight: bold; color: #58a6ff; margin: 20px 0 10px 0; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
    .aspect-card { background-color: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 15px; margin: 8px 0; }
    .positive { color: #3fb950; font-size: 18px; font-weight: bold; }
    .negative { color: #f85149; font-size: 18px; font-weight: bold; }
    .confidence { color: #8b949e; font-size: 13px; }
    .model-bert { font-size: 16px; font-weight: bold; color: #58a6ff; text-align: center; padding: 10px; background: #161b22; border-radius: 8px; margin-bottom: 12px; }
    .model-deberta { font-size: 16px; font-weight: bold; color: #bc8cff; text-align: center; padding: 10px; background: #161b22; border-radius: 8px; margin-bottom: 12px; }
    .aspects-found { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 12px 18px; margin: 12px 0; }
    .stButton > button { background-color: #238636; color: white; font-weight: bold; border-radius: 6px; border: none; padding: 10px 20px; width: 100%; }
    .stButton > button:hover { background-color: #2ea043; }
    .stTextArea textarea { background-color: #161b22; color: #ffffff; border: 1px solid #21262d; border-radius: 6px; }
    .stSelectbox > div > div { background-color: #161b22; color: #ffffff; }
    div[data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    bert_tokenizer = BertTokenizer.from_pretrained('bert_model/best_model')
    bert_model = BertForSequenceClassification.from_pretrained('bert_model/best_model')
    bert_model = bert_model.to(device)
    bert_model.eval()
    deberta_tokenizer = AutoTokenizer.from_pretrained('deberta_model/best_model')
    deberta_model = AutoModelForSequenceClassification.from_pretrained('deberta_model/best_model')
    deberta_model = deberta_model.to(device)
    deberta_model.eval()
    nlp = spacy.load('en_core_web_sm')
    with open('discovered_aspects.json', 'r') as f:
        discovered_aspects = json.load(f)
    return bert_tokenizer, bert_model, deberta_tokenizer, deberta_model, nlp, discovered_aspects, device

@st.cache_resource
def load_explainer():
    return LimeTextExplainer(class_names=['Negative', 'Positive'])

def extract_aspects(text, nlp, discovered_aspects):
    text_lower = text.lower()
    found = [a for a in discovered_aspects if a in text_lower]
    if not found:
        doc = nlp(text[:500])
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ['NOUN', 'PROPN'] and not chunk.root.is_stop:
                found.append(chunk.text.lower())
    return list(set(found)) if found else ['general']

def predict_sentiment(text, aspect, tokenizer, model, device):
    input_text = f"[ASPECT] {aspect} [REVIEW] {text}"
    encoding = tokenizer(input_text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    sentiment = 'positive' if probs[1] > probs[0] else 'negative'
    confidence = float(max(probs))
    return sentiment, confidence

def get_lime_explanation(text, tokenizer, model, device, explainer):
    def predict_proba(texts):
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs
    exp = explainer.explain_instance(text, predict_proba, num_features=8, num_samples=50)
    return exp.as_list()

# sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    use_bert = st.checkbox("🔵 BERT", value=True)
    use_deberta = st.checkbox("🔴 deBERTa", value=True)
    show_lime = st.checkbox("🧠 LIME Explanation", value=True)
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    **🔵 BERT**  
    Accuracy: 93.34%  
    F1: 0.9336  
    Kappa: 0.7918  
    
    **🔴 deBERTa**  
    Accuracy: 94.67%  
    F1: 0.9469  
    Kappa: 0.8332
    """)
    st.markdown("---")
    st.markdown("""
    <div style='color:#8b949e; font-size:12px; text-align:center;'>
    Charles Darwin University<br>NLP Project 2026
    </div>
    """, unsafe_allow_html=True)

# main
st.markdown('<div class="title">🎮 Steam Review ABSA System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aspect-Based Sentiment Analysis using BERT and deBERTa</div>', unsafe_allow_html=True)

with st.spinner("Loading models..."):
    bert_tokenizer, bert_model, deberta_tokenizer, deberta_model, nlp, discovered_aspects, device = load_models()
    explainer = load_explainer()

st.success("✅ Models loaded and ready!")

if 'history' not in st.session_state:
    st.session_state.history = []

examples = [
    "Write your own...",
    "The gameplay is absolutely fantastic but the graphics look outdated and performance is terrible",
    "Amazing story and great characters but the price is way too high for what you get",
    "Multiplayer is broken and full of bugs but the combat mechanics are really smooth",
    "The controls feel clunky but the world design is breathtaking and music is incredible",
    "Great fps shooter with smooth combat but the story is shallow and dlc is overpriced"
]

st.markdown('<div class="section">📝 Review Input</div>', unsafe_allow_html=True)

selected_example = st.selectbox("Choose an example or write your own", examples)
review_text = st.text_area(
    "Enter Steam game review",
    value="" if selected_example == "Write your own..." else selected_example,
    height=120,
    placeholder="e.g. The gameplay is amazing but the graphics are terrible..."
)

col1, col2 = st.columns([3, 1])
with col1:
    analyse_button = st.button("🔍 Analyse Review")
with col2:
    clear_button = st.button("🗑️ Clear History")

if clear_button:
    st.session_state.history = []
    st.success("History cleared!")

if analyse_button and review_text.strip():
    st.session_state.history.append(review_text[:80] + "..." if len(review_text) > 80 else review_text)

    st.markdown("---")

    with st.spinner("Extracting aspects..."):
        aspects = extract_aspects(review_text, nlp, discovered_aspects)

    st.markdown(f"""
    <div class="aspects-found">
        <b style="color:#8b949e; font-size:13px;">ASPECTS DETECTED</b><br>
        <span style="font-size:16px; color:#ffffff;">{' &nbsp;•&nbsp; '.join([a.upper() for a in aspects])}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section">🤖 Model Predictions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    bert_results = {}
    deberta_results = {}

    with col1:
        st.markdown('<div class="model-bert">🔵 BERT</div>', unsafe_allow_html=True)
        if use_bert:
            for aspect in aspects:
                sentiment, confidence = predict_sentiment(review_text, aspect, bert_tokenizer, bert_model, device)
                bert_results[aspect] = (sentiment, confidence)
                emoji = "✅" if sentiment == 'positive' else "❌"
                sent_class = "positive" if sentiment == 'positive' else "negative"
                st.markdown(f"""
                <div class="aspect-card">
                    <b style="color:#8b949e; font-size:12px;">{aspect.upper()}</b><br>
                    <span class="{sent_class}">{emoji} {sentiment.capitalize()}</span><br>
                    <span class="confidence">Confidence: {confidence*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("BERT disabled")

    with col2:
        st.markdown('<div class="model-deberta">🔴 deBERTa</div>', unsafe_allow_html=True)
        if use_deberta:
            for aspect in aspects:
                sentiment, confidence = predict_sentiment(review_text, aspect, deberta_tokenizer, deberta_model, device)
                deberta_results[aspect] = (sentiment, confidence)
                emoji = "✅" if sentiment == 'positive' else "❌"
                sent_class = "positive" if sentiment == 'positive' else "negative"
                st.markdown(f"""
                <div class="aspect-card">
                    <b style="color:#8b949e; font-size:12px;">{aspect.upper()}</b><br>
                    <span class="{sent_class}">{emoji} {sentiment.capitalize()}</span><br>
                    <span class="confidence">Confidence: {confidence*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("deBERTa disabled")

    st.markdown('<div class="section">📊 Comparison Summary</div>', unsafe_allow_html=True)

    summary_data = []
    for aspect in aspects:
        bert_sent, bert_conf = bert_results.get(aspect, ('N/A', 0))
        deb_sent, deb_conf = deberta_results.get(aspect, ('N/A', 0))
        agree = bert_sent == deb_sent
        summary_data.append({
            'Aspect': aspect.upper(),
            'BERT': f"{'✅' if bert_sent == 'positive' else '❌'} {bert_sent.capitalize()}",
            'BERT Conf': f"{bert_conf*100:.1f}%",
            'deBERTa': f"{'✅' if deb_sent == 'positive' else '❌'} {deb_sent.capitalize()}",
            'deBERTa Conf': f"{deb_conf*100:.1f}%",
            'Agreement': '🟢 Yes' if agree else '🔴 No'
        })

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    csv = pd.DataFrame(summary_data).to_csv(index=False)
    st.download_button("📥 Download Results as CSV", data=csv,
                      file_name="absa_results.csv", mime="text/csv")

    if show_lime and use_bert:
        st.markdown('<div class="section">🧠 LIME Explainability (BERT)</div>', unsafe_allow_html=True)
        st.caption("Shows which words pushed the model toward positive or negative sentiment")

        with st.spinner("Generating LIME explanation..."):
            word_weights = get_lime_explanation(review_text, bert_tokenizer, bert_model, device, explainer)

        words = [w[0] for w in word_weights]
        weights = [w[1] for w in word_weights]
        colors = ['#3fb950' if w > 0 else '#f85149' for w in weights]

        fig = go.Figure(go.Bar(
            x=weights,
            y=words,
            orientation='h',
            marker_color=colors,
            opacity=0.85
        ))
        fig.update_layout(
            xaxis=dict(title='Contribution to Positive Sentiment', color='#8b949e', gridcolor='#21262d'),
            yaxis=dict(color='#8b949e', gridcolor='#21262d'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#161b22',
            font=dict(color='#8b949e'),
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.history:
        st.markdown('<div class="section">📋 Review History</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #21262d; border-radius:6px; 
                        padding:8px 14px; margin:4px 0; color:#8b949e; font-size:13px;">
            {len(st.session_state.history)-i}. {item}
            </div>
            """, unsafe_allow_html=True)

elif analyse_button and not review_text.strip():
    st.warning("⚠️ Please enter a review before analysing!")

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:12px;'>
Charles Darwin University • NLP Project 2026 • Steam Reviews ABSA • BERT vs deBERTa
</div>
""", unsafe_allow_html=True)