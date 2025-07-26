from flask import Flask, request, jsonify, render_template_string
from better_profanity import profanity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load lightweight DialoGPT model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load profanity filter
profanity.load_censor_words()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MindCompanion ✨</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
            background: linear-gradient(to right, #141e30, #243b55);
            font-family: 'Arial', sans-serif;
            color: #fff;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
        }
        h1 {
            font-family: 'Great Vibes', cursive;
            font-size: 3em;
            margin-top: 30px;
        }
        #chatbox {
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            width: 90%;
            max-width: 600px;
            height: 60vh;
            overflow-y: auto;
            margin: 20px 0;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
        }
        .user {
            background-color: #3b5998;
            align-self: flex-end;
        }
        .bot {
            background-color: #8b9dc3;
            align-self: flex-start;
        }
        form {
            display: flex;
            width: 90%;
            max-width: 600px;
        }
        input {
            flex: 1;
            padding: 10px;
            border-radius: 10px;
            border: none;
            font-size: 1em;
        }
        button {
            padding: 10px 15px;
            background: #5a6e90;
            color: white;
            border: none;
            border-radius: 10px;
            margin-left: 10px;
            cursor: pointer;
        }
        button:hover {
            background: #3f4f66;
        }
    </style>
</head>
<body>
    <h1>MindCompanion</h1>
    <div id="chatbox"></div>
    <form onsubmit="sendMessage(event)">
        <input type="text" id="user_input" placeholder="What's on your mind?" required>
        <button type="submit">Send</button>
    </form>
<script>
    const chatbox = document.getElementById("chatbox");

    function addMessage(text, sender) {
        const msg = document.createElement("div");
        msg.className = "message " + sender;
        msg.textContent = text;
        chatbox.appendChild(msg);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    async function sendMessage(event) {
        event.preventDefault();
        const input = document.getElementById("user_input");
        const text = input.value;
        addMessage(text, "user");
        input.value = "";
        addMessage("Typing...", "bot");

        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        });

        const data = await res.json();
        const botMessages = document.querySelectorAll(".bot");
        botMessages[botMessages.length - 1].textContent = data.reply;
    }
</script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    if profanity.contains_profanity(user_input):
        return jsonify({"reply": "Let’s keep this a respectful space 🌸"})

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
    reply_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
