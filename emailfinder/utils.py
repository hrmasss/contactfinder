import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from domainfinder.utils import _clean_domain_text


def _clean_email(email: str) -> str:
    """Clean and validate email address"""
    if not email:
        return ""

    email = email.strip().lower()

    # Basic email validation
    if "@" not in email or email.count("@") != 1:
        return ""

    try:
        local, domain = email.split("@")
        domain = _clean_domain_text(domain)

        # Clean local part
        local = re.sub(r"[^\w\.\-\+]", "", local)

        if not local or not domain:
            return ""

        return f"{local}@{domain}"
    except Exception:
        return ""
