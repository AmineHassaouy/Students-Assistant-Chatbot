import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, request, jsonify, render_template, session
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

# ================================
# 🔧 Load API key + settings
# ================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("🚨 ERROR: Add GOOGLE_API_KEY to your .env file.")

from google.genai import Client, types

# Initialize Google GenAI client
client = Client(api_key=GOOGLE_API_KEY)

# Choose the Gemini model you want
MODEL_NAME = "gemini-2.5-flash"

# ================================
# 🔍 Chroma Vector Search
# ================================
TOP_K = 4
CHROMA_DIR = "db/chroma_db"

def search(query):
    results = vectorstore.similarity_search(query, k=TOP_K)
    return [(doc.metadata.get("source", "doc"), doc.page_content) for doc in results]

# ================================
# 🤖 Google GenAI LLM
# ================================
def format_prompt(history, context, user_msg):
    prompt_lines = ["You are a helpful assistant. Use the context below:"]
    if context:
        prompt_lines.append(f"Context:\n{context}")
    for role, msg in history:
        prompt_lines.append(f"{role}: {msg}")
    prompt_lines.append(f"user: {user_msg}")
    return "\n".join(prompt_lines)

def call_llm(prompt_text):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                max_output_tokens=2048,
                temperature=0.2
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ API error:\n{str(e)}"

# ================================
# 🚀 Load Chroma Vectorstore
# ================================
print("\n📦 Loading Chroma vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
print("💾 Vectorstore ready!")

# Per-session chat history (keyed by session ID)
# Capped at MAX_SESSIONS to prevent unbounded memory growth
user_histories = {}
MAX_HISTORY = 20   # turns kept per session
MAX_SESSIONS = 200  # max concurrent sessions stored in memory

# ================================
# 🌐 Flask Web App
# ================================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index1.html")

@app.post("/chat")
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"reply": "💬 Please type something..."}), 400

    # Get or create per-user history
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    sid = session["session_id"]

    if sid not in user_histories:
        # Evict oldest session if at capacity
        if len(user_histories) >= MAX_SESSIONS:
            oldest = next(iter(user_histories))
            del user_histories[oldest]
        user_histories[sid] = []

    chat_history = user_histories[sid]

    retrieved = search(question)
    context = "\n\n".join([f"[{src}]\n{txt}" for src, txt in retrieved])

    prompt = format_prompt(chat_history, context, question)
    reply = call_llm(prompt)

    # Store history, keep only last MAX_HISTORY turns
    chat_history.append(("user", question))
    chat_history.append(("assistant", reply))
    if len(chat_history) > MAX_HISTORY * 2:
        user_histories[sid] = chat_history[-(MAX_HISTORY * 2):]

    return jsonify({"reply": reply})

# ================================
# ▶️ Run Server
# ================================
if __name__ == "__main__":
    app.run(debug=True)
