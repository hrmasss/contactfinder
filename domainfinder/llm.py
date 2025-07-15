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
    company_name: str, domains: List[dict], llm_manager: LLMManager
) -> List[dict]:
    if len(domains) <= 3:
        return domains

    domain_list = "\n".join(
        [f"- {d['domain']}: {d['email_count']} emails" for d in domains[:15]]
    )

    prompt = f"""
    Company: "{company_name}"
    
    Email domains found:
    {domain_list}
    
    Filter to keep only domains DIRECTLY related to "{company_name}".
    
    Remove:
    - Generic domains (gmail, yahoo, hotmail)
    - Unrelated companies
    - Partner/vendor domains
    - Personal domains
    
    Keep:
    - Primary company domains
    - Subsidiary domains
    - Department domains
    
    Return ONLY a JSON array of relevant domain names:
    ["domain1.com", "domain2.org"]
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

        filtered = [d for d in domains if d["domain"] in relevant_domains]
        return filtered if filtered else domains
    except Exception:
        return domains
