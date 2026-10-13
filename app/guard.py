"""Guardrails, run BEFORE retrieval/LLM so blocked input costs nothing.
Order matters: self-harm gets a supportive reply (never an 'offensive content' refusal)."""
import re
from openai import OpenAI

SELF_HARM = re.compile(
    r"\b(i|i'm|im|i am)\b.{0,40}\b(suicid\w*|kill myself|end my life|want to die|hurt myself|self[- ]?harm|don'?t want to live)",
    re.I)
PROBE = re.compile(
    r"(system (prompt|instruction|setting)s?|ignore (all |your )?(previous|prior) instructions|"
    r"reveal your (prompt|instructions)|developer message|jailbreak)", re.I)

SUPPORT = ("I'm really sorry you're feeling this way, and I'm glad you said it. You don't have to go through "
           "this alone. If you might act on these thoughts or are in immediate danger, please contact your local "
           "emergency number now, or reach out to someone you trust and stay with them. You can find a crisis "
           "line for your country at https://findahelpline.com. If you'd like, you can tell me what's going on; "
           "I'm happy to listen.")
PROBE_REPLY = ("I can't share private system instructions or internal configuration. I can answer questions, "
               "summarize documents, and cite claims from your selected sources.")
MOD_REPLY = "I can't help with that request. Please keep questions focused on your sources and notes."

def check(text: str):
    """Return a canned reply if the message must not go to the LLM, else None."""
    if SELF_HARM.search(text):
        return SUPPORT
    if PROBE.search(text):
        return PROBE_REPLY
    try:  # OpenAI moderation endpoint is free; fail open if unavailable
        r = OpenAI().moderations.create(model="omni-moderation-latest", input=text).results[0]
        if r.flagged:
            return SUPPORT if (r.categories.self_harm or r.categories.self_harm_intent) else MOD_REPLY
    except Exception:
        pass
    return None
