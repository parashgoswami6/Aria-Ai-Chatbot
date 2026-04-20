"""
model.py — NLP Engine for AI Chatbot
Uses NLTK for text preprocessing + TF-IDF + Cosine Similarity for intent matching
Falls back gracefully on missing dependencies
"""

import json
import os
import random
import re
import logging
from typing import Tuple, Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Lazy imports with fallbacks ────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus   import stopwords
    from nltk.stem     import WordNetLemmatizer

    # Download required NLTK data silently
    for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    NLTK_AVAILABLE = True
    logger.info("NLTK loaded successfully")
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available — using basic tokenization")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise        import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn loaded successfully")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available — using keyword matching")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
class TextPreprocessor:
    """Handles text cleaning, tokenization, and lemmatization."""

    def __init__(self):
        self.lemmatizer  = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.stop_words  = set(stopwords.words("english")) if NLTK_AVAILABLE else set()
        # Keep question words — important for intent detection
        self.stop_words -= {"what", "who", "where", "when", "why", "how", "which"}

    def clean(self, text: str) -> str:
        """Lowercase, remove punctuation, extra spaces."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> list:
        """Tokenize using NLTK or basic split."""
        text = self.clean(text)
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        return text.split()

    def lemmatize(self, tokens: list) -> list:
        """Reduce words to base form: running → run."""
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def remove_stopwords(self, tokens: list) -> list:
        """Remove common words that add little meaning."""
        if self.stop_words:
            return [t for t in tokens if t not in self.stop_words]
        return tokens

    def process(self, text: str) -> str:
        """Full pipeline: clean → tokenize → lemmatize → remove stopwords."""
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
class IntentClassifier:
    """
    TF-IDF based intent classifier.
    Falls back to keyword matching if sklearn unavailable.
    """

    CONFIDENCE_THRESHOLD = 0.25   # Minimum similarity score to accept a match

    def __init__(self, intents: list):
        self.intents     = intents
        self.preprocessor = TextPreprocessor()
        self.vectorizer  = None
        self.corpus      = []      # Processed patterns
        self.intent_map  = []      # (tag, original_pattern) for each corpus entry

        self._build_corpus()
        if SKLEARN_AVAILABLE:
            self._fit_vectorizer()

    def _build_corpus(self):
        """Build training corpus from all intent patterns."""
        for intent in self.intents:
            if intent["tag"] == "fallback":
                continue
            for pattern in intent["patterns"]:
                processed = self.preprocessor.process(pattern)
                if processed:
                    self.corpus.append(processed)
                    self.intent_map.append(intent["tag"])

        logger.info(f"Corpus built: {len(self.corpus)} patterns across {len(self.intents)} intents")

    def _fit_vectorizer(self):
        """Fit TF-IDF vectorizer on the corpus."""
        if not self.corpus:
            return
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),    # Unigrams + bigrams
            analyzer="word",
            min_df=1,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        logger.info("TF-IDF vectorizer fitted")

    def predict_tfidf(self, text: str) -> Tuple[str, float]:
        """Use TF-IDF + cosine similarity to find best matching intent."""
        processed = self.preprocessor.process(text)
        if not processed:
            return "fallback", 0.0

        query_vec  = self.vectorizer.transform([processed])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx   = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.CONFIDENCE_THRESHOLD:
            return self.intent_map[best_idx], best_score
        return "fallback", best_score

    def predict_keyword(self, text: str) -> Tuple[str, float]:
        """Simple keyword matching fallback."""
        text_lower = text.lower()
        best_tag   = "fallback"
        best_score = 0.0

        for intent in self.intents:
            if intent["tag"] == "fallback":
                continue
            for pattern in intent["patterns"]:
                pattern_words = pattern.lower().split()
                matches = sum(1 for w in pattern_words if w in text_lower)
                score   = matches / max(len(pattern_words), 1)
                if score > best_score:
                    best_score = score
                    best_tag   = intent["tag"]

        return (best_tag, best_score) if best_score > 0.3 else ("fallback", 0.0)

    def predict(self, text: str) -> Tuple[str, float]:
        """Main prediction — uses TF-IDF if available, else keyword."""
        if SKLEARN_AVAILABLE and self.vectorizer:
            return self.predict_tfidf(text)
        return self.predict_keyword(text)


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class ResponseGenerator:
    """Selects appropriate response for a given intent tag."""

    def __init__(self, intents: list):
        self.intent_responses = {
            intent["tag"]: intent["responses"]
            for intent in intents
        }

    def get_response(self, tag: str) -> str:
        """Return a random response for the given tag."""
        responses = self.intent_responses.get(tag)
        if not responses:
            responses = self.intent_responses.get("fallback", ["I'm not sure about that."])
        return random.choice(responses)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHATBOT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class ChatBot:
    """
    Main chatbot class.
    Orchestrates: text preprocessing → intent classification → response generation
    """

    INTENTS_PATH = os.path.join(os.path.dirname(__file__), "data", "intents.json")

    def __init__(self):
        self.intents    = self._load_intents()
        self.classifier = IntentClassifier(self.intents)
        self.generator  = ResponseGenerator(self.intents)
        self.preprocessor = TextPreprocessor()
        logger.info("ChatBot engine initialized ✅")

    def _load_intents(self) -> list:
        """Load and validate intents from JSON file."""
        try:
            with open(self.INTENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            intents = data.get("intents", [])
            logger.info(f"Loaded {len(intents)} intents from {self.INTENTS_PATH}")
            return intents
        except FileNotFoundError:
            logger.error(f"intents.json not found at {self.INTENTS_PATH}")
            return self._default_intents()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in intents file: {e}")
            return self._default_intents()

    def _default_intents(self) -> list:
        """Minimal fallback intents if file not found."""
        return [
            {"tag": "greeting", "patterns": ["hello", "hi"], "responses": ["Hello!"]},
            {"tag": "fallback", "patterns": [], "responses": ["I'm not sure about that."]}
        ]

    def get_response(self, user_message: str) -> dict:
        """
        Process user message and return structured response dict.

        Returns:
            {
                "response": str,      — Bot reply
                "intent":   str,      — Detected intent tag
                "confidence": float,  — Match confidence (0-1)
                "method":   str       — Which NLP method was used
            }
        """
        if not user_message or not user_message.strip():
            return {
                "response": "Please type something so I can help you!",
                "intent": "empty",
                "confidence": 1.0,
                "method": "validation"
            }

        # Sanitize input
        user_message = user_message.strip()[:500]   # Max 500 chars

        # Classify intent
        tag, confidence = self.classifier.predict(user_message)

        # Generate response
        response = self.generator.get_response(tag)

        method = "tfidf" if (SKLEARN_AVAILABLE and self.classifier.vectorizer) else "keyword"

        logger.info(f"Message: '{user_message[:50]}' → Intent: '{tag}' ({confidence:.2f}) via {method}")

        return {
            "response":   response,
            "intent":     tag,
            "confidence": round(confidence, 3),
            "method":     method
        }

    def reload_intents(self) -> bool:
        """Hot-reload intents from file (used by admin panel)."""
        try:
            self.intents    = self._load_intents()
            self.classifier = IntentClassifier(self.intents)
            self.generator  = ResponseGenerator(self.intents)
            logger.info("Intents reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload intents: {e}")
            return False


# ── Singleton instance ─────────────────────────────────────────────────────────
_chatbot_instance: Optional[ChatBot] = None

def get_chatbot() -> ChatBot:
    """Return shared ChatBot singleton."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = ChatBot()
    return _chatbot_instance
