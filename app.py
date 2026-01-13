import streamlit as st
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from groq import Groq
from dotenv import load_dotenv
# -----------------------------
# Setup
# -----------------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Industry Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# UI Header
# -----------------------------
st.title("📊 Industry-Based Sentiment Analysis Platform")
st.caption("Real-time ML + GenAI sentiment intelligence powered by Groq")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

industry = st.sidebar.selectbox(
    "Select Industry",
    ["Finance", "Healthcare", "E-commerce", "Education", "Social Media"]
)

mode = st.sidebar.radio(
    "Analysis Type",
    ["Quick Sentiment", "Detailed Business Insight (GenAI)"]
)

# -----------------------------
# User Input
# -----------------------------
text = st.text_area(
    "Enter customer feedback / review",
    height=180,
    placeholder="Example: The app crashes frequently after the latest update..."
)

analyze = st.button("🔍 Analyze")

# -----------------------------
# Helper Functions
# -----------------------------
def sentiment_label(score):
    if score >= 0.05:
        return "Positive 😀"
    elif score <= -0.05:
        return "Negative 😡"
    else:
        return "Neutral 😐"

def groq_insight(text, industry, sentiment):
    prompt = f"""
You are a senior business analyst.

Industry: {industry}
Detected Sentiment: {sentiment}

Customer Feedback:
\"\"\"{text}\"\"\"

Provide:
1. Root cause of the sentiment
2. Business risk or opportunity
3. One actionable recommendation

Be professional, concise, and industry-focused.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# -----------------------------
# Analysis Logic
# -----------------------------
if analyze and text.strip():

    scores = sia.polarity_scores(text)
    sentiment = sentiment_label(scores["compound"])

    st.subheader("📌 Sentiment Scores")

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", scores["pos"])
    c2.metric("Neutral", scores["neu"])
    c3.metric("Negative", scores["neg"])

    st.subheader("🧠 Overall Sentiment")
    st.success(sentiment)

    if mode == "Detailed Business Insight (GenAI)":
        with st.spinner("Generating business insight using Groq..."):
            insight = groq_insight(text, industry, sentiment)

        st.subheader("📈 AI-Generated Business Insight")
        st.write(insight)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🚀 ML + GenAI Sentiment Platform | Powered by Groq LPU")
