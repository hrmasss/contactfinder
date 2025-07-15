import re
from .imports import clean_domain


def clean_email(email: str) -> str:
    """Clean and validate email address"""
    if not email:
        return ""

    email = email.strip().lower()

    # Basic email validation
    if "@" not in email or email.count("@") != 1:
        return ""

    try:
        local, domain = email.split("@")
        domain = clean_domain(domain)

        # Clean local part
        local = re.sub(r"[^\w\.\-\+]", "", local)

        if not local or not domain:
            return ""

        return f"{local}@{domain}"
    except Exception:
        return ""
