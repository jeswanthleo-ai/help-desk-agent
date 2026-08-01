import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def send_escalation_email(query: str, agent_name: str):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipient = os.getenv("SUPPORT_TEAM_EMAIL")

    body = f"""Help Desk Escalation — Unresolved Query

Timestamp: {datetime.now().isoformat()}
Agent Attempted: {agent_name}
User's Question: {query}

KB Search Summary: No sufficiently relevant content found above the
confidence threshold.

Suggested Action: Please review and update the KB or respond to the
user directly.
"""
    msg = MIMEText(body)
    msg["Subject"] = f"[Help Desk Escalation] Unresolved Query - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print("Escalation email sent.")
    except Exception as e:
        print(f"Email failed (non-fatal for demo): {e}")