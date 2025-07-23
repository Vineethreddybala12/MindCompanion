from flask import Flask, request, jsonify, render_template_string
from better_profanity import profanity
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

app = Flask(__name__)

# Load BlenderBot model
print("Loading BlenderBot model...")
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded successfully!")

# Load profanity filter
profanity.load_censor_words()

# 🌌 Starlit Sanctuary HTML
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MindCompanion ✨</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=EB+Garamond&display=swap" rel="stylesheet">
    <style>
        body {
            margin: 0;
            background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
            height: 100vh;
            font-family: 'EB Garamond', serif;
            color: white;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            font-size: 2.5em;
            margin-top: 30px;
        }
        h2 {
            font-size: 1.2em;
            font-style: italic;
            color: #ccc;
        }
        .stars {
            background: url('https://raw.githubusercontent.com/VincentGarreau/particles.js/master/demo/media/stars.png') repeat;
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            animation: moveStars 200s linear infinite;
        }
        @keyframes moveStars {
            from {background-position: 0 0;}
            to {background-position: -10000px 5000px;}
        }
        .chatbox {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 20px;
            width: 90%;
            max-width: 600px;
            height: 60vh;
            overflow-y: auto;
            margin: 20px 0;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        }
        .message {
            display: flex;
            margin: 12px 0;
            align-items: flex-end;
        }
        .message.bot {
            justify-content: flex-start;
        }
        .message.user {
            justify-content: flex-end;
        }
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
        .form-container {
            display: flex;
            width: 90%;
            max-width: 600px;
            margin-bottom: 20px;
        }
        input[type=text] {
            flex: 1;
            padding: 12px;
            border-radius: 12px;
            border: none;
            font-size: 1em;
            outline: none;
        }
        button {
            padding: 12px 18px;
            background: #5a6e90;
            border: none;
            color: white;
            border-radius: 12px;
            margin-left: 10px;
            cursor: pointer;
        }
        button:hover {
            background: #455671;
        }
    </style>
</head>
<body>
    <div class="stars"></div>
    <h1>🤖MINDCOMPANION</h1>
    <h2>-Reflect, Release, Renew</h2>
    <div class="chatbox" id="chatbox"></div>
    <form class="form-container" onsubmit="sendMessage(event)">
        <input type="text" id="user_input" placeholder="Share what's on your mind..." required>
        <button type="submit">Send</button>
    </form>
<script>
    const chatbox = document.getElementById('chatbox');

    function addMessage(content, sender) {
        const msg = document.createElement('div');
        msg.className = 'message ' + sender;
        msg.innerHTML = `
            ${sender === 'bot' ? '<img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png"/>' : ''}
            <div class="bubble">${content}</div>
            ${sender === 'user' ? '<img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png"/>' : ''}
        `;
        chatbox.appendChild(msg);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    async function sendMessage(event) {
        event.preventDefault();
        const input = document.getElementById('user_input');
        const text = input.value;
        addMessage(text, 'user');
        input.value = '';
        addMessage('Typing...', 'bot');

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text})
        });

        const data = await response.json();
        const botMessages = document.querySelectorAll('.bot .bubble');
        botMessages[botMessages.length - 1].innerText = data.reply;
    }
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    if profanity.contains_profanity(user_input):
        return jsonify({"reply": "Let’s keep this a safe and kind place 🚫"})

    inputs = tokenizer([user_input], return_tensors="pt").to(device)
    reply_ids = model.generate(**inputs)
    reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)