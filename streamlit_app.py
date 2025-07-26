import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DialoGPT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None
if 'past_inputs' not in st.session_state:
    st.session_state.past_inputs = []
if 'generated_responses' not in st.session_state:
    st.session_state.generated_responses = []

# Page config
st.set_page_config(page_title="MindCompanion ✨", layout="centered")
st.title("🤖 MindCompanion")
st.subheader("Reflect · Release · Renew")

# Dark/Light toggle
theme = st.toggle("🌗 Toggle Dark Mode")
if theme:
    st.markdown(
        """
        <style>
        body { background-color: #f5f5f5; color: #222; }
        .stTextInput input { background-color: #fff; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        body { background-color: #0d1b2a; color: #fff; }
        .stTextInput input { background-color: #1f4068; color: #fff; }
        </style>
        """, unsafe_allow_html=True)

# Chatbox display
for i in range(len(st.session_state.past_inputs)):
    st.markdown(f"**🧑 You:** {st.session_state.past_inputs[i]}")
    st.markdown(f"**🤖 MindCompanion:** {st.session_state.generated_responses[i]}")

# User input
user_input = st.text_input("How are you feeling today?", key="input")

if st.button("Send") and user_input:
    # Encode user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append to chat history
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate reply
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    # Decode and store response
    reply = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.past_inputs.append(user_input)
    st.session_state.generated_responses.append(reply)
    st.experimental_rerun()
