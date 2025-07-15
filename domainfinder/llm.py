from typing import List, Dict, Any
from .schema import CompanyInfo
from .utils import LLMManager


def research_company(
    company_query: str, llm_manager: LLMManager, context: Dict[str, Any] = None
) -> CompanyInfo:
    # Build context section if provided
    context_section = ""
    if context:
        context_section = "\n\nAdditional context to identify the correct company:\n"
        for key, value in context.items():
            if value:
                context_section += f"- {key.title()}: {value}\n"

    prompt = f"""
    Research "{company_query}" and provide ONLY the JSON response below.
    {context_section}
    For "{company_query}", return exactly this JSON format:
    {{
        "company_name": "Short searchable name",
        "website": "primary-domain.com", 
        "likely_email_domains": ["primary.com", "alternative.com"],
        "description": "Brief description"
    }}
    
    Return ONLY the JSON object, no other text.
    """

    response = llm_manager.query(prompt)

    if not response.success:
        return CompanyInfo(
            company_name=llm_manager.clean_company_name(company_query),
            website="",
            likely_email_domains=[],
            description="",
        )

    try:
        data = response.data
        if data.get("company_name"):
            data["company_name"] = llm_manager.clean_company_name(data["company_name"])
        return CompanyInfo(**data)
    except Exception:
        return CompanyInfo(
            company_name=llm_manager.clean_company_name(company_query),
            website="",
            likely_email_domains=[],
            description="",
        )


def filter_relevant_domains(
    company_name: str,
    domains: List[dict],
    llm_manager: LLMManager,
    company_info: CompanyInfo = None,
) -> List[dict]:
    if len(domains) <= 3:
        return domains

    domain_list = "\n".join(
        [f"- {d['domain']}: {d['email_count']} emails" for d in domains[:15]]
    )

    # Include company research context
    research_context = ""
    if company_info:
        research_context = f"""
Company Research Context:
- Official Company: {company_info.company_name}
- Known Website: {company_info.website}
- Expected Domains: {", ".join(company_info.likely_email_domains)}
- Description: {company_info.description}
"""

    prompt = f"""
    Company: "{company_name}"
    {research_context}
    
    Raw domains found (may contain malformed domains):
    {domain_list}
    
    Tasks:
    1. Clean up malformed domains (e.g., "example.comasdfsadf" → "example.com")
    2. Keep only domains DIRECTLY related to "{company_name}"
    3. Use the company research context to make informed decisions
    
    REMOVE:
    - Generic domains (gmail, yahoo, hotmail, outlook)
    - Unrelated companies
    - Personal domains
    - Domains with completely different business focus
    
    KEEP AND CLEAN:
    - Primary company domains (clean up if malformed)
    - Official website domain and its variations
    - Subsidiary domains
    - Department/division domains
    - Regional domains (compound TLDs like .co.uk, .edu.bd)
    
    Return ONLY a JSON array of cleaned, relevant domain names:
    ["cleaneddomain1.com", "cleaneddomain2.co.uk"]
    """

    response = llm_manager.query(prompt)

    if not response.success:
        return domains

    try:
        data = response.data
        if isinstance(data, list):
            relevant_domains = data
        else:
            for value in data.values():
                if isinstance(value, list):
                    relevant_domains = value
                    break
            else:
                return domains

        # Consolidate cleaned domains and merge their data
        domain_map = {}
        for original_domain in domains:
            # Find if this domain has a cleaned version
            cleaned_version = None
            for clean_domain in relevant_domains:
                if clean_domain.replace(".", "").replace("-", "") in original_domain[
                    "domain"
                ].replace(".", "").replace("-", ""):
                    cleaned_version = clean_domain
                    break

            if cleaned_version:
                if cleaned_version not in domain_map:
                    domain_map[cleaned_version] = {
                        "domain": cleaned_version,
                        "email_count": 0,
                        "source_count": 0,
                        "confidence": 0.0,
                        "from_llm": False,
                        "mx_valid": False,
                    }

                # Merge data from all versions of this domain
                domain_map[cleaned_version]["email_count"] += original_domain[
                    "email_count"
                ]
                domain_map[cleaned_version]["source_count"] += original_domain[
                    "source_count"
                ]
                domain_map[cleaned_version]["from_llm"] = (
                    domain_map[cleaned_version]["from_llm"]
                    or original_domain["from_llm"]
                )
                domain_map[cleaned_version]["mx_valid"] = (
                    domain_map[cleaned_version]["mx_valid"]
                    or original_domain["mx_valid"]
                )

        # Recalculate confidence scores after consolidation
        for domain_data in domain_map.values():
            confidence = 0.0
            if domain_data["from_llm"]:
                confidence += 0.4  # Higher weight for LLM suggestions
            if domain_data["email_count"] > 0:
                confidence += min(0.4, domain_data["email_count"] * 0.05)
            if domain_data["source_count"] > 0:
                confidence += min(0.3, domain_data["source_count"] * 0.03)

            # Bonus for likely official domains
            if any(
                word in domain_data["domain"] for word in company_name.lower().split()
            ):
                confidence += 0.2

            domain_data["confidence"] = round(min(1.0, confidence), 3)

        return list(domain_map.values()) if domain_map else domains
    except Exception:
        return domains
