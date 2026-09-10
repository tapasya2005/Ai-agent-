import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Groq client — Compound (open-source reasoning model via Groq Cloud)
# Priority: environment variable > .env file > hardcoded fallback
# ---------------------------------------------------------------------------
_FALLBACK_KEY = "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"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", _FALLBACK_KEY)
GROQ_MODEL    = os.getenv("GROQ_MODEL", "groq/compound-mini")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# System prompt — smart farming domain expert
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI agricultural advisor specialising in smart farming.
Your role is to help farmers and agronomists with:

- Crop selection and rotation strategies
- Soil health, fertility, and amendment recommendations
- Irrigation and water management advice
- Pest and disease identification and integrated pest management (IPM)
- Weather impact analysis and climate-resilient farming techniques
- Precision farming, IoT sensors, and data-driven decisions
- Fertiliser scheduling and organic/bio alternatives
- Harvesting, post-harvest handling, and storage best practices
- Sustainable and regenerative farming practices
- Market timing and profitability guidance for smallholders

Always give practical, evidence-based advice that is actionable for both
smallholder and commercial farmers. When you need more context (soil type,
location, crop stage, season) ask concisely. Format responses clearly using
bullet points or numbered steps where appropriate. Keep answers focused,
helpful, and encouraging."""

# ---------------------------------------------------------------------------
# In-memory conversation history keyed by session_id
# ---------------------------------------------------------------------------
conversation_store: dict[str, list[dict]] = {}


def build_messages(session_id: str, user_message: str) -> list[dict]:
    """Append the new user turn and return the full message list."""
    history = conversation_store.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def call_groq(messages: list[dict]) -> str:
    """Call the Groq API and return the assistant reply text."""
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as exc:
        return f"⚠️ Groq API error: {exc}"


# ---------------------------------------------------------------------------
# Static farming knowledge shown in the sidebar
# ---------------------------------------------------------------------------
QUICK_TIPS = [
    {"icon": "🌱", "title": "Soil Testing",    "tip": "Test soil pH and nutrients every season for optimal yield."},
    {"icon": "💧", "title": "Irrigation",       "tip": "Drip irrigation reduces water use by up to 50 % vs flood methods."},
    {"icon": "🐛", "title": "Pest Control",     "tip": "Integrated Pest Management (IPM) minimises chemical inputs."},
    {"icon": "🌤️", "title": "Weather Watch",    "tip": "Monitor 7-day forecasts before planting or applying inputs."},
    {"icon": "🔄", "title": "Crop Rotation",    "tip": "Rotate legumes with cereals to naturally fix nitrogen."},
    {"icon": "📊", "title": "Yield Tracking",   "tip": "Record yields per plot to spot performance trends over time."},
]

EXAMPLE_QUESTIONS = [
    "What cover crops improve soil health in sandy loam?",
    "How do I identify early blight on tomatoes?",
    "Best irrigation schedule for wheat in semi-arid regions?",
    "How can I reduce fertiliser costs without losing yield?",
    "What are signs of nitrogen deficiency in maize?",
    "How do I manage aphids organically on brassicas?",
]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        quick_tips=QUICK_TIPS,
        example_questions=EXAMPLE_QUESTIONS,
        model_name=GROQ_MODEL,
    )


@app.route("/chat", methods=["POST"])
def chat():
    data         = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    session_id   = (data.get("session_id") or "default").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    messages = build_messages(session_id, user_message)
    reply    = call_groq(messages)

    # Store assistant reply in history
    conversation_store[session_id].append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply, "session_id": session_id})


@app.route("/reset", methods=["POST"])
def reset():
    data       = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "default").strip()
    conversation_store.pop(session_id, None)
    return jsonify({"status": "ok"})


@app.route("/health")
def health():
    """Lightweight health-check endpoint."""
    try:
        # A minimal test call to verify the API key and model are reachable
        test = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        groq_ok = bool(test.choices[0].message.content)
    except Exception:
        groq_ok = False

    return jsonify({
        "flask": "ok",
        "groq":  "ok" if groq_ok else "unreachable",
        "model": GROQ_MODEL,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
