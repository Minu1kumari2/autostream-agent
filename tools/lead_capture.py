"""
Tool: Lead Capture
Simulates a backend API call to capture a qualified lead.
Only invoked after Name, Email, and Platform are all collected.
"""

import re
from datetime import datetime, timezone


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.

    In a real deployment this would POST to a CRM (HubSpot, Salesforce, etc.).

    Args:
        name:     Full name of the lead.
        email:    Email address of the lead.
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        A dict with status and a confirmation message.
    """
    # Basic validation
    if not name or not email or not platform:
        raise ValueError("All three fields (name, email, platform) are required.")

    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(email_pattern, email):
        raise ValueError(f"Invalid email format: {email}")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Simulated print output (as required by assignment spec)
    print(f"Lead captured successfully: {name}, {email}, {platform}")

    return {
        "status": "success",
        "message": f"Lead captured successfully: {name}, {email}, {platform}",
        "lead": {
            "name": name,
            "email": email,
            "platform": platform,
            "captured_at": timestamp,
            "source": "AutoStream Social Agent",
        },
    }
