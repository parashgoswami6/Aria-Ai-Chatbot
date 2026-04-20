"""
app.py — Flask Server for AI Chatbot
Routes: Chat API, History API, Admin Panel, Voice (optional)
"""

import os
import uuid
import json
import logging
from datetime import datetime
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template,
    session, redirect, url_for, flash
)

from model    import get_chatbot
from database import get_db

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "aria-chatbot-secret-2024")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Admin credentials (change in production / use env vars)
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")

# Initialize NLP model and database at startup
chatbot = get_chatbot()
db      = get_db()

logger.info("Flask app initialized ✅")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_session_id() -> str:
    """Get or create a unique session ID for the user."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def admin_required(f):
    """Decorator: redirect to login if not authenticated as admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated

def success_response(data: dict, status: int = 200):
    return jsonify({"status": "success", **data}), status

def error_response(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    """Serve the main chat UI."""
    session_id = get_session_id()
    db.create_session(session_id)
    return render_template("index.html", session_id=session_id)


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body: { "message": "user text" }
    Returns: { "response": "...", "intent": "...", "confidence": 0.9 }
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response("Invalid JSON body", 400)

        user_msg = data.get("message", "").strip()
        if not user_msg:
            return error_response("Message cannot be empty", 400)

        if len(user_msg) > 500:
            return error_response("Message too long (max 500 characters)", 400)

        session_id = get_session_id()

        # Process with NLP model
        result = chatbot.get_response(user_msg)

        # Save to database
        db.save_exchange(
            session_id  = session_id,
            user_msg    = user_msg,
            bot_msg     = result["response"],
            intent      = result["intent"],
            confidence  = result["confidence"]
        )

        return success_response({
            "response":   result["response"],
            "intent":     result["intent"],
            "confidence": result["confidence"],
            "method":     result["method"],
            "timestamp":  datetime.utcnow().strftime("%H:%M")
        })

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return error_response("Internal server error. Please try again.", 500)


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    GET /api/history
    Returns chat history for current session.
    """
    try:
        session_id = get_session_id()
        history    = db.get_history(session_id)
        return success_response({"history": history, "session_id": session_id})
    except Exception as e:
        logger.error(f"History error: {e}")
        return error_response("Failed to retrieve history", 500)


@app.route("/api/clear", methods=["POST"])
def clear_history():
    """
    POST /api/clear
    Clear current session's chat history.
    """
    try:
        session_id = session.pop("session_id", None)
        if session_id:
            db.clear_session(session_id)
        session["session_id"] = str(uuid.uuid4())
        return success_response({"message": "Chat history cleared"})
    except Exception as e:
        logger.error(f"Clear error: {e}")
        return error_response("Failed to clear history", 500)


@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    """Return quick-reply suggestions for the UI."""
    suggestions = [
        "Hello! How are you?",
        "What can you do?",
        "Tell me a joke 😄",
        "What is machine learning?",
        "Explain Python",
        "Motivate me!",
        "What is NLP?",
        "How does Flask work?"
    ]
    return success_response({"suggestions": suggestions})


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Admin login page."""
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USER and password == ADMIN_PASS:
            session["admin_logged_in"] = True
            logger.info(f"Admin login from {request.remote_addr}")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials. Try again.", "error")

    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    """Admin dashboard with stats and history."""
    stats      = db.get_stats()
    sessions   = db.get_all_sessions()
    all_msgs   = db.get_all_history(limit=100)

    # Load current intents for display
    intents_path = os.path.join(os.path.dirname(__file__), "data", "intents.json")
    intents = []
    try:
        with open(intents_path) as f:
            intents = json.load(f).get("intents", [])
    except Exception:
        pass

    return render_template("admin.html",
        stats    = stats,
        sessions = sessions,
        messages = all_msgs,
        intents  = intents
    )


@app.route("/admin/intents", methods=["GET"])
@admin_required
def admin_intents():
    """Return intents JSON for editing."""
    intents_path = os.path.join(os.path.dirname(__file__), "data", "intents.json")
    try:
        with open(intents_path) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return error_response(str(e), 500)


@app.route("/admin/intents", methods=["POST"])
@admin_required
def admin_update_intents():
    """Update intents.json and reload the model."""
    intents_path = os.path.join(os.path.dirname(__file__), "data", "intents.json")
    try:
        data = request.get_json(silent=True)
        if not data or "intents" not in data:
            return error_response("Invalid intents format", 400)

        # Validate structure
        for intent in data["intents"]:
            if not all(k in intent for k in ["tag", "patterns", "responses"]):
                return error_response("Each intent must have tag, patterns, responses", 400)

        # Save to file
        with open(intents_path, "w") as f:
            json.dump(data, f, indent=2)

        # Reload model
        success = chatbot.reload_intents()
        if success:
            return success_response({"message": "Intents updated and model reloaded!"})
        else:
            return error_response("Saved but model reload failed", 500)

    except Exception as e:
        logger.error(f"Intent update error: {e}")
        return error_response(str(e), 500)


@app.route("/admin/stats")
@admin_required
def admin_stats():
    """Return JSON stats for admin panel."""
    return success_response(db.get_stats())


# ══════════════════════════════════════════════════════════════════════════════
# VOICE SUPPORT (Optional)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/voice/stt", methods=["POST"])
def speech_to_text():
    """
    POST /api/voice/stt
    Receives audio blob, returns transcribed text using SpeechRecognition.
    """
    try:
        import speech_recognition as sr
        import io

        audio_data = request.files.get("audio")
        if not audio_data:
            return error_response("No audio file provided", 400)

        recognizer = sr.Recognizer()
        audio_bytes = audio_data.read()

        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return success_response({"text": text})

    except ImportError:
        return error_response("Speech recognition not installed. Run: pip install SpeechRecognition", 501)
    except Exception as e:
        logger.error(f"STT error: {e}")
        return error_response(f"Speech recognition failed: {str(e)}", 500)


@app.route("/api/voice/tts", methods=["POST"])
def text_to_speech():
    """
    POST /api/voice/tts
    Converts text to speech using pyttsx3 (basic) or gTTS.
    Returns audio URL or base64.
    Note: Browser's built-in SpeechSynthesis API is used in the frontend instead.
    """
    return success_response({
        "message": "TTS is handled client-side via Web Speech API",
        "supported": True
    })


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return error_response("Endpoint not found", 404)
    return render_template("index.html"), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {e}")
    if request.path.startswith("/api/"):
        return error_response("Internal server error", 500)
    return render_template("index.html"), 500


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"

    logger.info(f"Starting Aria Chatbot on http://localhost:{port}")
    logger.info(f"Admin panel: http://localhost:{port}/admin  (admin / admin123)")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        use_reloader=False    # Prevent double model loading
    )
