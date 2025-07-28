import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import re
import nltk

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Offensive words list
OFFENSIVE_WORDS = {"idiot", "stupid", "dumb", "hate", "useless", "worthless"}

# Check for offensive language
def is_offensive(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return any(word in OFFENSIVE_WORDS for word in words)

# Load Blenderbot model + tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-1B-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Get bot response
def get_bot_reply(user_input):
    if is_offensive(user_input):
        return "I'm here to support you. Let's keep this space respectful and kind 💙"
    inputs = tokenizer([user_input], return_tensors="pt")
    result = model.generate(**inputs)
    return tokenizer.decode(result[0], skip_special_tokens=True)

# ---- Streamlit App ---- #
st.set_page_config(page_title="MindCompanion ✨", layout="centered")

# HTML + CSS for frontend
st.markdown("""
    <style>
        body {
            margin: 0;
            background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
            height: 100vh;
            font-family: 'EB Garamond', serif;
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .chatbox {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 20px;
            width: 90%;
            max-width: 600px;
            height: 60vh;
            overflow-y: auto;
            margin: 20px auto;
            box-shadow: 0 0 20px rgba(255,255,255,0.1);
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
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 style='text-align:center;'>MindCompanion</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;font-style:italic;color:#ccc;'>\"Reflect, Release, Renew\"</h3>", unsafe_allow_html=True)
st.markdown("<div class='chatbox'>", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    role = msg["role"]
    bubble = f"<div class='message {role}'><div class='bubble'>{msg['content']}</div></div>"
    st.markdown(bubble, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input form
with st.form("chat_input_form", clear_on_submit=True):
    user_input = st.text_input("Share what's on your mind...", key="user_input")
    submitted = st.form_submit_button("Send")

# Process input
if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    bot_reply = get_bot_reply(user_input)
    st.session_state.messages.append({"role": "bot", "content": bot_reply})
    
