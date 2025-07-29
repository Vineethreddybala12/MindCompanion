import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import nltk
import re

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Basic profanity filter
OFFENSIVE_WORDS = {"idiot", "stupid", "dumb", "hate", "useless", "worthless"}

def is_offensive(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return any(word in OFFENSIVE_WORDS for word in words)

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Generate bot reply
def get_bot_reply(user_input):
    if is_offensive(user_input):
        return "I'm here to support you. Let's keep things respectful and kind 💙"

    inputs = tokenizer([user_input], return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=100)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Streamlit page setup
st.set_page_config(page_title="MindCompanion ✨", layout="centered")

# Styling
st.markdown("""
<style>
    body {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
        color: white;
        font-family: 'EB Garamond', serif;
    }
    .chatbox {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 20px;
        max-width: 600px;
        margin: 20px auto;
        height: 60vh;
        overflow-y: auto;
    }
    .message {
        display: flex;
        margin: 12px 0;
        align-items: flex-end;
    }
    .message.bot { justify-content: flex-start; }
    .message.user { justify-content: flex-end; }
    .bubble {
        padding: 12px 18px;
        border-radius: 18px;
        max-width: 70%;
        line-height: 1.5;
    }
    .bot .bubble {
        background: #2a3b4c;
        color: #eee;
        margin-left: 10px;
    }
    .user .bubble {
        background: #3f6072;
        color: #fff;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>MindCompanion ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;font-style:italic;color:#ccc;'>\"Reflect, Release, Renew\"</h3>", unsafe_allow_html=True)
st.markdown("<div class='chatbox'>", unsafe_allow_html=True)

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    st.markdown(f"<div class='message {role}'><div class='bubble'>{content}</div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Share what's on your mind...")
    submitted = st.form_submit_button("Send")

# Handle submission
if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    bot_reply = get_bot_reply(user_input)
    st.session_state.messages.append({"role": "bot", "content": bot_reply})
    st.rerun()
