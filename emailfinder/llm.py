import re
import requests
from typing import List, Dict, Any
from .schema import EmailResult
from .utils import clean_email
from .imports import LLMManager


def research_employee_emails(
    company_name: str,
    employee_name: str,
    domains: List[str],
    llm_manager: LLMManager,
    employee_context: Dict[str, Any] = None,
    scraped_emails: List[str] = None,
    domain_subdomains: Dict[str, List[str]] = None,
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
        subdomains = domain_subdomains.get(domain, []) if domain_subdomains else []
        if subdomains:
            domains_section += f"   Subdomains: {', '.join(subdomains)}\n"
    domains_section += "\nNote: These domains were found through automated analysis. Verify relevance and conduct additional research if needed."

    prompt = f"""
Company: "{company_name}"
Employee: "{employee_name}"
{context_section}
{domains_section}
{scraped_section}

TASK: Find the most likely email addresses for this employee.

SUBDOMAIN USAGE:
- If subdomains are listed for a domain, prioritize using the appropriate subdomain based on employee context
- Match employee location/department to relevant subdomain (e.g., us.company.com for US employees)
- If no employee context matches, use the most common subdomain or main domain

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
    "email": "john.doe@us.company.com",
    "confidence": 0.85,
    "pattern_type": "public",
    "source": "company directory",
    "domain": "us.company.com",
    "pattern_frequency": 1
  }}
]

LIMITS:
- MUST return at least 1 email (use fallback patterns if needed)
- Return 1 email if absolutely certain (publicly found)
- Return 2-5 emails if based on company patterns
- Return up to 15 emails if using common patterns
- Maximum 15 emails

FALLBACK PATTERNS (use if no better data available):
- firstname.lastname@domain.com
- firstname@domain.com
- lastname@domain.com
- f.lastname@domain.com
- firstnamelastname@domain.com

Focus on QUALITY over quantity. Better to return fewer, more likely emails than many uncertain ones. NEVER return empty results.
"""

    response = llm_manager.query(prompt)

    if not response.success:
        # Fallback: generate basic patterns when LLM fails
        return generate_fallback_emails(employee_name, domains[:1])

    try:
        data = response.data
        if isinstance(data, list):
            result = data
        elif isinstance(data, dict):
            # Look for email arrays in the response
            result = []
            for key, value in data.items():
                if isinstance(value, list) and value:
                    result = value
                    break
        else:
            result = []

        # Ensure we have at least one result
        if not result:
            result = generate_fallback_emails(employee_name, domains[:1])

        return result
    except Exception:
        return generate_fallback_emails(employee_name, domains[:1])


def generate_fallback_emails(
    employee_name: str, domains: List[str]
) -> List[Dict[str, Any]]:
    """Generate fallback email patterns when LLM fails"""
    if not domains:
        return []

    domain = domains[0]
    name_parts = employee_name.lower().split()

    if len(name_parts) < 2:
        return []

    first, last = name_parts[0], name_parts[-1]

    patterns = [
        f"{first}.{last}@{domain}",
        f"{first}@{domain}",
        f"{last}@{domain}",
        f"{first[0]}{last}@{domain}",
        f"{first}{last[0]}@{domain}",
    ]

    return [
        {
            "email": pattern,
            "confidence": 0.3,
            "pattern_type": "fallback",
            "source": "system_generated",
            "domain": domain,
            "pattern_frequency": 1,
        }
        for pattern in patterns
    ]


def scrape_employee_emails(
    company_name: str,
    employee_name: str,
    domains: List[str],
    serper_api_key: str,
    max_results: int = 5,
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
                # Handle encoding properly
                response.encoding = response.apparent_encoding or "utf-8"
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
                        cleaned = clean_email(email)
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
            cleaned_email = clean_email(email)
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
