# 🤖 Aria — Smart AI Chatbot

A production-ready AI chatbot with NLP, web interface, REST API, admin panel, and voice support.

---

## 📁 Project Structure

```
chatbot/
├── app.py              ← Flask server + all routes
├── model.py            ← NLP engine (TF-IDF + NLTK)
├── database.py         ← SQLite chat history manager
├── requirements.txt    ← Dependencies
├── data/
│   └── intents.json    ← Knowledge base (22 intents, 200+ patterns)
├── templates/
│   ├── index.html      ← Main chat UI
│   ├── admin.html      ← Admin dashboard
│   └── admin_login.html
└── static/
    └── style.css       ← Complete styling (dark/light theme)
```

---

## ⚡ Quick Setup (5 minutes)

### Step 1 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
python app.py
```

### Step 4 — Open in browser
```
Chat UI:      http://localhost:5000
Admin Panel:  http://localhost:5000/admin
Admin Login:   admin / admin123
```

---

## 🔧 Tech Stack

| Layer      | Technology              |
|------------|-------------------------|
| Backend    | Python 3.10+ / Flask    |
| NLP        | NLTK + scikit-learn     |
| Algorithm  | TF-IDF + Cosine Similarity |
| Database   | SQLite (auto-created)   |
| Frontend   | HTML5, CSS3, Vanilla JS |
| Voice      | Web Speech API (browser)|

---

## 🧠 How NLP Works

```
User Input
    ↓
Text Preprocessing (NLTK)
  - Lowercase + clean
  - Tokenize (word_tokenize)
  - Lemmatize (WordNetLemmatizer)
  - Remove stopwords
    ↓
TF-IDF Vectorization
  - Transform to numeric vector
  - Compare with all training patterns
    ↓
Cosine Similarity
  - Find most similar intent
  - Confidence threshold: 0.25
    ↓
Response Selection
  - Random response from matched intent
  - Fallback if confidence too low
    ↓
Save to SQLite → Return JSON
```

---

## 🎯 Features

- ✅ **22 Intent categories** — greetings, tech, jokes, motivation, and more
- ✅ **TF-IDF + Cosine Similarity** — intelligent NLP matching
- ✅ **Real-time responses** — async fetch, no page reload
- ✅ **SQLite history** — every message saved with intent + confidence
- ✅ **Admin panel** — live stats, message history, edit intents JSON
- ✅ **Voice input** — Web Speech API (Chrome supported)
- ✅ **Text-to-speech** — SpeechSynthesis API
- ✅ **Dark/Light theme** — persisted in localStorage
- ✅ **Session management** — UUID-based per-user sessions
- ✅ **Responsive design** — mobile friendly

---

## 🔑 API Reference

### POST /api/chat
```json
Request:  { "message": "Hello!" }
Response: {
  "status":     "success",
  "response":   "Hello! How can I help you today?",
  "intent":     "greeting",
  "confidence": 0.87,
  "method":     "tfidf",
  "timestamp":  "14:32"
}
```

### GET /api/history
```json
Response: {
  "status": "success",
  "history": [
    { "role": "user", "content": "Hello", "timestamp": "..." },
    { "role": "bot",  "content": "Hi!", "intent": "greeting", ... }
  ]
}
```

### POST /api/clear
```json
Response: { "status": "success", "message": "Chat history cleared" }
```

---

## ➕ Adding Custom Intents

Edit `data/intents.json`:
```json
{
  "tag": "your_topic",
  "patterns": [
    "ask about topic",
    "question about subject",
    "how does X work"
  ],
  "responses": [
    "Here's what I know about X...",
    "Great question! X works by..."
  ],
  "context": ""
}
```

Then reload via Admin Panel → Manage Intents → Save & Reload Model

---

## 🌐 Environment Variables

```bash
SECRET_KEY=your-secret-key    # Flask session secret
ADMIN_USER=your-username      # Admin panel username
ADMIN_PASS=your-password      # Admin panel password
PORT=5000                     # Server port
DEBUG=false                   # Disable for production
```

---

## 🚀 Deploy to Cloud

### Heroku
```bash
echo "web: python app.py" > Procfile
heroku create your-chatbot-name
git push heroku main
```

### Railway / Render
- Connect GitHub repo
- Set PORT environment variable
- Deploy automatically

---

## 🎤 Voice Features

- **Voice Input**: Click the microphone button → speak → auto-sends
- **Voice Output**: Click the speaker icon to toggle TTS responses
- **Browser Support**: Chrome, Edge (full support), Firefox (partial)

---

## 📊 Admin Panel Features

| Feature | Description |
|---------|-------------|
| Dashboard | Total messages, sessions, top intents |
| Chat History | All conversations with intent labels |
| Manage Intents | Edit/add intents without restart |
| Sessions | Active session overview |

Admin: `http://localhost:5000/admin` → `admin / admin123`
