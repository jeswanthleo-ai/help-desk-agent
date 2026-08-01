import os
import re
import random
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options=types.HttpOptions(
        timeout=15000,  # 15s per attempt
        retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1, max_delay=5),
    ),
)

# --- Fast path: exact greetings/thanks/farewells, no LLM call needed ---

GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|hiya|yo|good\s+(morning|afternoon|evening))[\s!.,]*$", re.IGNORECASE)
THANKS_RE = re.compile(
    r"^\s*(many\s+)?(thanks?|thank\s*you|thankyou|ty|thx|appreciate it|cheers)"
    r"(\s+(a\s+lot|a\s+bunch|so\s+much|very\s+much|so\s+very\s+much))?[\s!.,]*$",
    re.IGNORECASE,
)
FAREWELL_RE = re.compile(r"^\s*(bye|goodbye|see\s+(you|ya)|take care|later)[\s!.,]*$", re.IGNORECASE)

# Identity/small-talk phrases — not exact one-word greetings, so these use
# search() over the whole message instead of a full-string match.
SMALLTALK_RE = re.compile(
    r"how\s+are\s+you|"
    r"who\s+are\s+you|"
    r"what'?s?\s+your\s+name|"
    r"what\s+is\s+your\s+name|"
    r"what\s+(can|do)\s+you\s+do|"
    r"what\s+are\s+your\s+capabilities|"
    r"how\s+can\s+you\s+help( me)?|"
    r"what\s+can\s+you\s+help\s+(me\s+)?with",
    re.IGNORECASE,
)

CASUAL_RESPONSES = {
    "greeting": [
        "Hey there! 👋 What can I help you with today?",
        "Hi! How can I help you today?",
    ],
    "thanks": [
        "You're welcome! Let me know if there's anything else you need.",
        "Anytime! Happy to help further if you need it.",
    ],
    "farewell": [
        "Take care! Reach out anytime you need help.",
        "Bye! Feel free to come back if you have more questions.",
    ],
    "smalltalk": [
        "I'm doing great, thanks for asking! I'm the Help Desk Agent — I can "
        "help with platform/technical issues and account/access requests. "
        "What do you need help with?",
    ],
}

def classify_casual(query: str) -> str | None:
    """Fast regex check for greetings/thanks/farewells/small-talk so they skip the LLM entirely."""
    q = query.strip()
    if GREETING_RE.match(q):
        return "greeting"
    if THANKS_RE.match(q):
        return "thanks"
    if FAREWELL_RE.match(q):
        return "farewell"
    if SMALLTALK_RE.search(q):
        return "smalltalk"
    return None

def get_casual_response(kind: str) -> str:
    return random.choice(CASUAL_RESPONSES[kind])

# --- LLM path: freeform small talk that regex can't reliably catch ---
# (e.g. "hi how are you", "what's your name", "what can you do")

ROUTING_PROMPT = """You are a routing classifier for a helpdesk system.
Classify the user query into exactly one category:

- "platform": technical issues, errors, how-to questions, troubleshooting
- "account": access requests, permissions, onboarding, passwords, account issues
- "casual": greetings, small talk, thanks, farewells, or questions about the
  assistant itself (e.g. "how are you", "what's your name", "what can you do",
  "how can you help me") — anything that isn't an actual support request

Respond with ONLY the single word "platform", "account", or "casual". No explanation.

User query: {query}
"""

CASUAL_RESPONSE_PROMPT = """You are the friendly front-desk agent for a Help Desk
Assistant. Reply casually and briefly (1-2 sentences) to the message below.

If asked what you are, your name, or what you can help with, mention you're the
Help Desk Agent and can help with platform/technical issues (errors, how-tos,
troubleshooting) and account/access requests (passwords, permissions, onboarding,
VPN, Jira, software installs).

User message: {query}

Response:
"""

def classify_intent(query: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=ROUTING_PROMPT.format(query=query)
        )
    except (errors.APIError, httpx.TimeoutException, httpx.ConnectError):
        return "platform"  # API unavailable — safe default

    intent = response.text.strip().lower()
    if intent not in ("platform", "account", "casual"):
        intent = "platform"  # safe default
    return intent

def generate_casual_response(query: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=CASUAL_RESPONSE_PROMPT.format(query=query)
        )
        return response.text.strip()
    except (errors.APIError, httpx.TimeoutException, httpx.ConnectError):
        return (
            "Hey! I'm the Help Desk Agent — I can help with platform issues "
            "and account/access requests. What do you need?"
        )

if __name__ == "__main__":
    test_queries = [
        "How do I get Jira access?",
        "I'm seeing a 403 error on the dashboard",
        "What's the password policy?",
        "My data isn't syncing",
        "Hi how are you",
        "what is your name",
        "what are your capabilities",
    ]
    for q in test_queries:
        print(f"{q}  ->  {classify_intent(q)}")
