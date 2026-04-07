# EST Khénifra Chatbot

An AI-powered chatbot for the **École Supérieure de Technologie de Khénifra (EST Khénifra)**. It answers student and visitor questions about programs, departments, administrative procedures, clubs, and more — in French and Arabic — using a Retrieval-Augmented Generation (RAG) pipeline backed by Google Gemini.

---

## Screenshots

### Desktop

![Chat interface with annotations](designs/chat-with-annotations.png)

### Tablet / Mobile

![Tablet mockup](designs/tablette-mockup.png)

---

## How It Works

```
User question
     │
     ▼
ChromaDB vector search  ──►  Top-K relevant document chunks
     │
     ▼
Google Gemini 2.5 Flash  (context + history + question)
     │
     ▼
Answer streamed back to the browser
```

1. **Ingestion (`ingest.py`)** — reads `.txt` files from `docs/`, splits them into chunks, embeds them with `all-MiniLM-L6-v2`, and persists the vector store to `db/chroma_db/`.
2. **Serving (`chat.py`)** — a Flask app that retrieves the top-4 relevant chunks per question, builds a prompt with per-session chat history, and calls Gemini to generate the answer.
3. **Frontend (`templates/index1.html`)** — a responsive single-page UI with a sticky navbar, info sections, and a floating chat widget.

---

## Project Structure

```
chatbot/
├── chat.py               # Flask app + RAG chat logic
├── ingest.py             # Document ingestion & embedding
├── requirements.txt
├── .env                  # API keys (not committed)
├── docs/                 # Knowledge base (.txt files)
│   ├── est_khenifra.txt
│   ├── Inscriptions.txt
│   ├── Informations générales.txt
│   ├── Démarches Administratives.txt
│   ├── Clubs.txt
│   └── ...               # One file per department / topic
├── db/
│   └── chroma_db/        # Persisted vector store (auto-generated)
├── static/
│   └── images/           # Logo, icons, and other assets
├── templates/
│   └── index1.html       # Main UI template
└── designs/              # UI mockups & design references
    ├── chat-with-annotations.png
    └── tablette-mockup.png
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd chatbot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
SECRET_KEY=any_random_secret_string
```

### Ingest Documents

Run this once (or whenever you update the `docs/` folder):

```bash
python ingest.py
```

This creates the `db/chroma_db/` vector store. Re-running it will clear and rebuild it from scratch.

### Run the App

```bash
python chat.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Knowledge Base

The `docs/` folder contains the information the chatbot can answer questions about:

| File | Topic |
|------|-------|
| `est_khenifra.txt` | General school information |
| `Informations générales.txt` | General info & contacts |
| `Inscriptions.txt` | Enrollment & registration |
| `Démarches Administratives.txt` | Administrative procedures |
| `Clubs.txt` | Student clubs |
| `BTV.txt` | BTV department |
| `GE.txt` | Génie Électrique |
| `GME.txt` | Génie Mécanique et Énergétique |
| `GTER.txt` | Génie des Travaux et Équipements Ruraux |
| `IDIA.txt` | Ingénierie des Données et Intelligence Artificielle |
| `ILR.txt` | ILR department |
| `MRH.txt` | Management des Ressources Humaines |
| `TAA.txt` | Techniques Administratives et Artistiques |
| `TCC.txt` | TCC department |
| `E-AMO.txt` | E-AMO program |

To add new information, create a `.txt` file in `docs/` and re-run `python ingest.py`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | Flask |
| LLM | Google Gemini 2.5 Flash (via `google-genai`) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace / SentenceTransformers) |
| Vector store | ChromaDB |
| RAG framework | LangChain |
| Frontend | Vanilla HTML/CSS/JS + Marked.js |

---

## Configuration Options

| Variable (in `chat.py`) | Default | Description |
|-------------------------|---------|-------------|
| `TOP_K` | `4` | Number of document chunks retrieved per query |
| `MAX_HISTORY` | `20` | Conversation turns kept in memory per session |
| `MAX_SESSIONS` | `200` | Max concurrent sessions held in memory |
| `MODEL_NAME` | `gemini-2.5-flash` | Gemini model used for generation |
