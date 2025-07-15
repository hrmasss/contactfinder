import re
import requests
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .schema import EmailResult
from .utils import _clean_email
from domainfinder.llm import LLMManager


def research_employee_emails(
    company_name: str,
    employee_name: str,
    domains: List[str],
    llm_manager: LLMManager,
    employee_context: Dict[str, Any] = None,
    scraped_emails: List[str] = None,
) -> List[Dict[str, Any]]:
    """Research employee emails using LLM with company and employee context"""

    # Build employee context section
    context_section = ""
    if employee_context:
        context_section = "\n\nEmployee context:\n"
        for key, value in employee_context.items():
            if value:
                context_section += f"- {key.title()}: {value}\n"

    # Build scraped emails section
    scraped_section = ""
    if scraped_emails:
        scraped_section = (
            "\n\nScraped emails from web (may contain garbage - filter carefully):\n"
        )
        for email in scraped_emails[:20]:
            scraped_section += f"- {email}\n"
        scraped_section += "\nNote: These emails were scraped from web sources and may contain irrelevant or outdated information. Use them as reference only."

    # Build domains section
    domains_section = (
        "\n\nTop company domains (from domain finder - not absolute truth):\n"
    )
    for i, domain in enumerate(domains[:3], 1):
        domains_section += f"{i}. {domain}\n"
    domains_section += "\nNote: These domains were found through automated analysis. Verify relevance and conduct additional research if needed."

    prompt = f"""
Company: "{company_name}"
Employee: "{employee_name}"
{context_section}
{domains_section}
{scraped_section}

TASK: Find the most likely email addresses for this employee.

PRIORITY RULES (in order):
1. PUBLICLY AVAILABLE EMAILS: If you find publicly available emails for this exact person at this company, return them with HIGH confidence (0.8-1.0)
2. COMPANY EMAIL PATTERNS: Research the company's email patterns using the provided domains, then generate emails based on discovered patterns with MEDIUM confidence (0.5-0.8)
3. COMMON PATTERNS: If insufficient data, use standard business email patterns with LOW confidence (0.2-0.5)

FILTERING RULES:
- REMOVE: Personal emails (gmail, yahoo, hotmail, etc.)
- REMOVE: Generic/role emails (info@, contact@, support@, etc.)
- REMOVE: Irrelevant domains not related to the company
- REMOVE: Clearly outdated or spam emails from scraped data
- KEEP: Only emails that could realistically belong to this specific employee

RESPONSE FORMAT:
Return a JSON array of email objects, sorted by confidence (highest first):
[
  {{
    "email": "john.doe@company.com",
    "confidence": 0.85,
    "pattern_type": "public",
    "source": "company directory",
    "domain": "company.com",
    "pattern_frequency": 1
  }}
]

LIMITS:
- Return 1 email if absolutely certain (publicly found)
- Return 2-5 emails if based on company patterns
- Return up to 15 emails if using common patterns
- Minimum 1 email, maximum 15 emails

Focus on QUALITY over quantity. Better to return fewer, more likely emails than many uncertain ones.
"""

    response = llm_manager.query(prompt)

    if not response.success:
        return []

    try:
        data = response.data
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Look for email arrays in the response
            for key, value in data.items():
                if isinstance(value, list) and value:
                    return value
        return []
    except Exception:
        return []


def scrape_employee_emails(
    company_name: str,
    employee_name: str,
    domains: List[str],
    serper_api_key: str,
    max_results: int = 20,
) -> List[str]:
    """Scrape web for employee emails"""

    # Generate search queries
    queries = []

    # Direct employee searches
    queries.extend(
        [
            f'"{employee_name}" "{company_name}" email',
            f'"{employee_name}" email site:linkedin.com',
            f'"{employee_name}" contact "{company_name}"',
        ]
    )

    # Domain-specific searches
    for domain in domains[:3]:
        queries.extend(
            [
                f'"{employee_name}" @{domain}',
                f'"{employee_name}" site:{domain}',
            ]
        )

    all_emails = set()

    for query in queries:
        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": serper_api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results // len(queries)},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("organic", [])

                for result in results:
                    # Extract emails from title, snippet, and link
                    text = f"{result.get('title', '')} {result.get('snippet', '')} {result.get('link', '')}"

                    # Find email patterns
                    email_pattern = (
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
                    )
                    found_emails = re.findall(email_pattern, text)

                    for email in found_emails:
                        cleaned = _clean_email(email)
                        if cleaned and any(domain in cleaned for domain in domains):
                            all_emails.add(cleaned)

        except Exception:
            continue

    return list(all_emails)


def filter_and_rank_emails(
    emails: List[Dict[str, Any]],
    domains: List[str],
) -> List[EmailResult]:
    """Filter and rank email results"""

    if not emails:
        return []

    results = []

    for email_data in emails:
        try:
            email = email_data.get("email", "")
            if not email or "@" not in email:
                continue

            # Clean email
            cleaned_email = _clean_email(email)
            if not cleaned_email:
                continue

            # Check if email domain matches our target domains
            email_domain = cleaned_email.split("@")[1]
            if email_domain not in domains:
                continue

            # Create EmailResult
            result = EmailResult(
                email=cleaned_email,
                confidence=email_data.get("confidence", 0.5),
                pattern_type=email_data.get("pattern_type", "unknown"),
                source=email_data.get("source", "llm"),
                domain=email_domain,
                pattern_frequency=email_data.get("pattern_frequency", 0),
            )

            results.append(result)

        except Exception:
            continue

    # Sort by confidence (highest first)
    results.sort(key=lambda x: x.confidence, reverse=True)

    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for result in results:
        if result.email not in seen:
            seen.add(result.email)
            unique_results.append(result)

    return unique_results
