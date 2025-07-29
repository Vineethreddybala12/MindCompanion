
AI Mental Health Chatbot using BlenderBot
# 🤖 MindCompanion – AI Mental Health Chatbot
                   -Reflect, Release, Renew
          
> An empathetic, supportive AI-powered chatbot built using BlenderBot and Streamlit to provide mental health support in a safe, kind, and conversational space.


## ✨ Features

- 🤖 Powered by **BlenderBot 400M (Distilled)** for contextual, human-like responses
- 💬 **Streamlit-based chat UI** with aesthetic frontend and smooth interactions
- 🛡 **Offensive language detection** to keep conversations safe
- 💡 Easy-to-use form for entering thoughts, feelings, or stressors
- 📜 Chat history maintained within a session


## 🚀 Live Demo

👉 [Launch MindCompanion on Streamlit] (https://mindcompanion12.streamlit.app)  


## 🛠 Tech Stack

- Python 🐍
- Streamlit
- HuggingFace Transformers
- Torch (PyTorch)
- NLTK (for offensive input detection)
  

## 📁 Project Structure
AI-Companion/
├── app.py # Main Streamlit app
├── requirements.txt # Required dependencies
└── README.md # Project documentation


## 🧠 How It Works

1. User types their thoughts in a chat input box.
2. Input is checked for offensive terms using regex + a word list.
3. If safe, it’s passed to the BlenderBot model via HuggingFace Transformers.
4. Bot responds empathetically, and both messages are shown in chat history.


## 📦 Installation (Run Locally)

### Prerequisites:

- Python 3.8+
- Git

### Steps:

```bash
git clone https://github.com/Vineethreddybala12/MindCompanion.git
cd MindCompanion
pip install -r requirements.txt
streamlit run app.py

📤 Deployment (Streamlit Cloud)

1.Push code to GitHub

2.Go to https://streamlit.io/cloud

3.Click New App

4.Fill in:

 Repository: Vineethreddybala12/MindCompanion

 Branch: master

 File: app.py
 
5.Click Deploy 🚀

📜 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with attribution.

.

🙌 Acknowledgments

 Facebook BlenderBot 400M-distill

 Streamlit

 Flaticon for UI icons


