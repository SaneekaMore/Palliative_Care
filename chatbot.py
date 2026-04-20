"""Emotional support chatbot utilities."""

from __future__ import annotations

from textblob import TextBlob


EMOTION_RESPONSES = {
    "Happy": "I'm so glad you're feeling positive. Keep nurturing those moments of comfort and connection.",
    "Neutral": "Thank you for sharing. I'm here with you—would you like a breathing exercise or a check-in prompt?",
    "Sad": "I'm really sorry you're feeling this way. You're not alone. A gentle step now could be calling someone you trust.",
    "Anxious": "That sounds overwhelming. Try slow breathing: inhale for 4, hold for 4, exhale for 6. I'm right here with you.",
}


def detect_emotion(message: str) -> str:
    """Map sentiment polarity to a simple emotional category."""
    polarity = TextBlob(message).sentiment.polarity

    if polarity > 0.25:
        return "Happy"
    if polarity < -0.35:
        return "Sad"
    if any(word in message.lower() for word in ["worried", "anxious", "scared", "panic", "stress"]):
        return "Anxious"
    return "Neutral"


def generate_support_response(message: str) -> tuple[str, str]:
    """Return detected emotion and a supportive response."""
    emotion = detect_emotion(message)
    prefix = {
        "Happy": "🌤️",
        "Neutral": "💬",
        "Sad": "💙",
        "Anxious": "🫶",
    }[emotion]

    response = f"{prefix} {EMOTION_RESPONSES[emotion]}"

    if "pain" in message.lower():
        response += " If your pain is increasing, please notify your care team promptly."
    if "alone" in message.lower():
        response += " Reaching out to a loved one now might help you feel supported."

    return emotion, response
