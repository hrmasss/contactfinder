import os
import re
import time
import json
import requests
import dns.resolver
from bs4 import BeautifulSoup
from collections import Counter
from typing import List, Dict, Set, Any, Optional
from pydantic import BaseModel, Field, field_validator

# LLM Imports
from google import genai
from google.genai import types
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities import GoogleSerperAPIWrapper


# ============================================================================
# COMMON UTILITIES
# ============================================================================


def clean_domain(domain: str) -> str:
    """Basic domain cleaning - keep it simple"""
    if not domain:
        return ""

    domain = domain.strip().lower()

    # Remove protocol prefixes
    if domain.startswith(("http://", "https://")):
        domain = domain.split("://", 1)[1]

    # Remove www prefix
    if domain.startswith("www."):
        domain = domain[4:]

    # Remove path, query, and fragment
    domain = domain.split("/")[0].split("?")[0].split("#")[0]

    # Remove common trailing separators and everything after
    domain = re.sub(r"[,;:!?\s].*$", "", domain)
    domain = re.sub(r"[(\[\{].*$", "", domain)
    domain = re.sub(r'["\'].*$', "", domain)

    # Basic validation
    if (
        len(domain) > 100
        or domain.startswith((".", "-"))
        or domain.endswith((".", "-"))
        or ".." in domain
        or ".-" in domain
        or "-." in domain
        or not domain
        or "." not in domain
    ):
        return ""

    return domain


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


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class CompanyInfo(BaseModel):
    company_name: str
    website: str = ""
    likely_email_domains: List[str] = Field(default_factory=list)
    description: str = ""
    sub_mail_domains: List[str] = Field(default_factory=list)

    @field_validator("likely_email_domains")
    @classmethod
    def validate_domains(cls, v):
        return [domain.strip().lower() for domain in v if domain.strip()]


class DomainResult(BaseModel):
    domain: str
    email_count: int = 0
    source_count: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    from_llm: bool = False
    mx_valid: bool = False
    sub_mail_domains: List[str] = Field(default_factory=list)


class EmployeeInfo(BaseModel):
    employee_name: str
    title: str = ""
    department: str = ""
    company_name: str = ""
    additional_context: str = ""


class EmailResult(BaseModel):
    email: str
    confidence: float = Field(ge=0.0, le=1.0)
    pattern_type: str = ""  # "public", "pattern", "guess"
    source: str = ""
    domain: str = ""
    pattern_frequency: int = 0  # How often this pattern appears

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower().strip()


class SearchResult(BaseModel):
    title: str = ""
    link: str
    snippet: str = ""


class LLMResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any] | None = None
    error: str | None = None
    provider: str


class ContactFinderResult(BaseModel):
    """Unified result schema for both company and employee search"""

    # Company information
    company_name: str
    website: str = ""
    likely_email_domains: List[str] = Field(default_factory=list)
    email_patterns: List[str] = Field(default_factory=list)  # Ranked patterns
    description: str = ""
    sub_mail_domains: List[str] = Field(default_factory=list)  # Ranked subdomains

    # Employee information (null if not searched)
    employee_name: Optional[str] = None
    employee_description: str = ""
    likely_email_addresses: List[EmailResult] = Field(
        default_factory=list
    )  # Ranked emails


# ============================================================================
# LLM AGENTS
# ============================================================================


def gemini(prompt: str) -> str:
    """Gemini 2.5 Flash with Google Search."""
    client = genai.Client()
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )
    return response.text


def gpt(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """GPT agent with search capabilities using langchain"""
    llm = init_chat_model(model, model_provider="openai", temperature=0)
    search = GoogleSerperAPIWrapper()

    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to search for information",
        )
    ]

    agent = create_react_agent(llm, tools)

    events = agent.stream(
        {"messages": [("user", prompt)]},
        stream_mode="values",
        config=RunnableConfig(recursion_limit=100),
    )

    # Get the last message from the agent
    for event in events:
        pass

    return event["messages"][-1].content


# ============================================================================
# CORE UTILITIES
# ============================================================================


class LLMManager:
    def __init__(self, provider_order: List[str] = None):
        self.provider_order = provider_order or ["gemini", "gpt"]
        self.providers = {}
        self._setup_providers()

    def _setup_providers(self):
        try:
            self.providers["gemini"] = gemini
        except Exception:
            pass

        try:
            self.providers["gpt"] = gpt
        except Exception:
            pass

        if not self.providers:
            raise RuntimeError("No LLM providers available")

    def _extract_json(self, text: str) -> dict:
        text = text.strip()

        if "```json" in text:
            json_part = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text and "{" in text:
            parts = text.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    json_part = part.strip()
                    break
            else:
                json_part = text
        else:
            # Try to find JSON object or array
            start_obj = text.find("{")
            start_arr = text.find("[")

            if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
                end = text.rfind("}")
                if end != -1 and end > start_obj:
                    json_part = text[start_obj : end + 1]
                else:
                    json_part = text
            elif start_arr != -1:
                end = text.rfind("]")
                if end != -1 and end > start_arr:
                    json_part = text[start_arr : end + 1]
                else:
                    json_part = text
            else:
                json_part = text

        json_part = json_part.replace("\n", " ").replace("\r", " ")
        json_part = re.sub(r",\s*}", "}", json_part)
        json_part = re.sub(r",\s*]", "]", json_part)

        try:
            result = json.loads(json_part)
            # If it's a list, wrap it in a dict
            if isinstance(result, list):
                return {"domains": result}
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty dict
            return {}

    def _call_provider(
        self, provider_name: str, provider_func, prompt: str, retries: int = 2
    ) -> LLMResponse:
        for attempt in range(retries):
            try:
                response = provider_func(prompt)
                data = self._extract_json(response)
                return LLMResponse(success=True, data=data, provider=provider_name)
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "overloaded" in error_msg.lower():
                    if attempt < retries - 1:
                        time.sleep((attempt + 1) * 3)
                        continue
                return LLMResponse(
                    success=False, error=error_msg, provider=provider_name
                )

        return LLMResponse(
            success=False, error="Max retries exceeded", provider=provider_name
        )

    def query(self, prompt: str, max_providers: int = 2) -> LLMResponse:
        providers_tried = 0

        for provider_name in self.provider_order:
            if providers_tried >= max_providers or provider_name not in self.providers:
                continue

            response = self._call_provider(
                provider_name, self.providers[provider_name], prompt
            )
            providers_tried += 1

            if response.success:
                print(f"Using {provider_name.upper()}")
                return response

            print(f"{provider_name.upper()} failed: {response.error}")
            time.sleep(1)

        return LLMResponse(success=False, error="All providers failed", provider="none")

    def clean_company_name(self, name: str) -> str:
        if not name:
            return name

        name = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", name)
        name = re.sub(r"\s*\([^)]*\d+[^)]*\)", "", name)
        name = re.sub(r"\s*\(owner and operator\)", "", name)
        name = re.sub(r"\s*\(owns [^)]*\)", "", name)
        name = re.sub(r"\s+", " ", name).strip()

        return name


class WebScraper:
    def __init__(self, serper_api_key: str):
        self.serper_api_key = serper_api_key
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def search_emails(
        self, company_name: str, max_results: int = 30
    ) -> List[SearchResult]:
        queries = [
            f'"{company_name}" email contact',
            f'"{company_name}" employees email',
            f'"{company_name}" staff directory',
            f'"{company_name}" @* contact',
            f"{company_name} email address",
            f"{company_name} contact information",
        ]

        all_results = []

        for query in queries:
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.serper_api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": max_results // len(queries)},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("organic", [])

                    for result in results:
                        all_results.append(
                            SearchResult(
                                title=result.get("title", ""),
                                link=result.get("link", ""),
                                snippet=result.get("snippet", ""),
                            )
                        )
            except Exception:
                continue

        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.link not in seen_urls:
                seen_urls.add(result.link)
                unique_results.append(result)

        return unique_results[:max_results]

    def search_employee_emails(
        self,
        company_name: str,
        employee_name: str,
        domains: List[str],
        max_results: int = 5,
    ) -> List[str]:
        """Search for specific employee emails"""
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
                        "X-API-KEY": self.serper_api_key,
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

    def extract_emails(self, text: str) -> Set[str]:
        """Extract and validate email addresses from text"""
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b"
        emails = set(re.findall(pattern, text, re.IGNORECASE))

        filtered = set()
        for email in emails:
            email = email.lower().strip()

            # Skip test/example emails
            if any(
                skip in email
                for skip in ["example.", "test@", "noreply", "no-reply", "placeholder"]
            ):
                continue

            if len(email) < 5 or "@" not in email:
                continue

            try:
                local_part, domain = email.split("@", 1)
                cleaned_domain = clean_domain(domain)

                if cleaned_domain and "." in cleaned_domain and len(cleaned_domain) > 3:
                    filtered.add(f"{local_part}@{cleaned_domain}")
            except ValueError:
                continue

        return filtered

    def scrape_page(self, url: str) -> Set[str]:
        """Scrape a webpage for email addresses"""
        try:
            response = requests.get(url, headers=self.headers, timeout=8)
            if response.status_code == 200:
                # Handle encoding properly
                response.encoding = response.apparent_encoding or "utf-8"
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract emails from page text
                text_emails = self.extract_emails(soup.get_text())

                # Extract emails from mailto links
                mailto_emails = set()
                for link in soup.find_all("a", href=re.compile(r"^mailto:")):
                    href = link.get("href", "")
                    if href.startswith("mailto:"):
                        email = (
                            href.replace("mailto:", "").split("?")[0].lower().strip()
                        )
                        if "@" in email:
                            mailto_emails.add(email)

                return text_emails.union(mailto_emails)
        except Exception:
            pass

        return set()

    def scrape_all(self, search_results: List[SearchResult]) -> Dict:
        """Scrape all search results and organize by domain"""
        all_emails = set()
        domain_sources = {}

        for result in search_results:
            emails = self.scrape_page(result.link)

            if emails:
                all_emails.update(emails)

                for email in emails:
                    if "@" in email:
                        try:
                            domain = email.split("@")[1]
                            cleaned_domain = clean_domain(domain)

                            if cleaned_domain and "." in cleaned_domain:
                                domain_sources.setdefault(cleaned_domain, []).append(
                                    result.link
                                )
                        except Exception:
                            continue

        return {"emails": list(all_emails), "domain_sources": domain_sources}


class DomainValidator:
    """Handles domain validation and DNS checks"""

    def has_mx_record(self, domain: str) -> bool:
        """Check if domain has valid MX or A records"""
        try:
            domain = clean_domain(domain)
            if not domain:
                return False

            try:
                mx_records = dns.resolver.resolve(domain, "MX")
                return len(mx_records) > 0
            except dns.resolver.NoAnswer:
                try:
                    a_records = dns.resolver.resolve(domain, "A")
                    return len(a_records) > 0
                except Exception:
                    return False
            except dns.resolver.NXDOMAIN:
                return False
        except Exception:
            return False

    def analyze_domains(
        self, scraped_data: Dict, llm_domains: List[str], llm_manager: LLMManager
    ) -> List[DomainResult]:
        """Analyze and rank domains using LLM for complex subdomain analysis"""
        emails = scraped_data["emails"]
        domain_sources = scraped_data["domain_sources"]

        # Use LLM to analyze domain patterns and subdomains
        domain_analysis = self._get_llm_domain_analysis(
            llm_manager, emails, llm_domains
        )

        # Build results from LLM analysis
        results = []
        for domain_info in domain_analysis:
            domain = domain_info.get("domain", "")
            if not domain:
                continue

            # Get email count for this domain
            email_count = sum(1 for email in emails if f"@{domain}" in email.lower())

            # Get source count
            source_count = len(domain_sources.get(domain, []))

            # Check if from original LLM suggestions
            from_llm = domain in llm_domains

            # Use confidence from LLM analysis
            confidence = min(1.0, domain_info.get("confidence", 0.5))

            # Get cleaned subdomains from LLM
            sub_domains = domain_info.get("sub_mail_domains", [])

            results.append(
                DomainResult(
                    domain=domain,
                    email_count=email_count,
                    source_count=source_count,
                    confidence=confidence,
                    from_llm=from_llm,
                    mx_valid=False,
                    sub_mail_domains=sub_domains,
                )
            )

        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def _get_llm_domain_analysis(
        self, llm_manager: LLMManager, emails: List[str], suggested_domains: List[str]
    ) -> List[Dict]:
        """Get LLM analysis of domain patterns and subdomains"""
        prompt = f"""
Analyze these email addresses and domain suggestions to identify the main company domains and their subdomains.

EMAIL ADDRESSES FOUND:
{chr(10).join(emails[:50])}  # Limit to first 50 for token efficiency

SUGGESTED DOMAINS FROM COMPANY RESEARCH:
{chr(10).join(suggested_domains)}

INSTRUCTIONS:
1. Group email domains by their main company domain (e.g., us.abb.com, ca.abb.com → main domain: abb.com)
2. Clean up malformed domains (remove test emails, invalid domains, etc.)
3. Identify relevant subdomains for each main domain
4. Calculate confidence scores based on:
   - Email frequency for this domain
   - Relevance to company (avoid generic domains like gmail.com)
   - Subdomain patterns indicating corporate structure
5. Only include domains that appear legitimate and company-related

Return a JSON array of domain objects:
```json
[
  {{
    "domain": "abb.com",
    "confidence": 0.95,
    "sub_mail_domains": ["us.abb.com", "ca.abb.com", "ch.abb.com"],
    "reasoning": "Main corporate domain with regional subdomains"
  }},
  {{
    "domain": "example.com", 
    "confidence": 0.3,
    "sub_mail_domains": [],
    "reasoning": "Secondary domain with few emails"
  }}
]
```

Focus on finding the TOP 5 most relevant domains. Exclude generic email providers (gmail, outlook, etc.).
"""

        response = llm_manager.query(prompt)

        if response.success and response.data:
            # Handle both direct array and nested data structure
            domain_data = response.data
            if isinstance(domain_data, dict) and "domains" in domain_data:
                return domain_data["domains"]
            elif isinstance(domain_data, list):
                return domain_data

        # Fallback to basic analysis if LLM fails
        return self._fallback_domain_analysis(emails, suggested_domains)

    def _fallback_domain_analysis(
        self, emails: List[str], suggested_domains: List[str]
    ) -> List[Dict]:
        """Fallback domain analysis when LLM fails"""
        domain_counts = Counter()
        subdomain_map = {}

        for email in emails:
            if "@" in email:
                try:
                    domain = email.split("@")[1]
                    domain = clean_domain(domain)
                    if domain:
                        domain_counts[domain] += 1

                        if domain.count(".") > 1:  # Has subdomain
                            main_domain = ".".join(domain.split(".")[-2:])
                            if main_domain not in subdomain_map:
                                subdomain_map[main_domain] = set()
                            subdomain_map[main_domain].add(domain)
                        else:
                            if domain not in subdomain_map:
                                subdomain_map[domain] = set()
                except Exception:
                    continue

        # Add suggested domains
        for domain in suggested_domains:
            domain = clean_domain(domain)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0)
                if domain not in subdomain_map:
                    subdomain_map[domain] = set()

        # Build results
        results = []
        for domain, email_count in domain_counts.items():
            confidence = min(1.0, 0.3 + (email_count * 0.01))
            sub_domains = list(subdomain_map.get(domain, set()))

            results.append(
                {
                    "domain": domain,
                    "confidence": confidence,
                    "sub_mail_domains": sub_domains,
                    "reasoning": f"Fallback analysis - {email_count} emails found",
                }
            )

        return sorted(results, key=lambda x: x["confidence"], reverse=True)[:5]


# ============================================================================
# LLM RESEARCH FUNCTIONS
# ============================================================================


def research_company(
    company_query: str, llm_manager: LLMManager, context: Dict[str, Any] = None
) -> CompanyInfo:
    """Research company information using LLM"""
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
        "description": "Brief description",
        "sub_mail_domains": ["us.domain.com", "ca.domain.com"]
    }}
    
    IMPORTANT: Only include sub_mail_domains if you find evidence that the company uses regional/departmental subdomains for email addresses. 
    
    Research the company's email structure to identify if they use patterns like:
    - Regional subdomains (us.company.com, ca.company.com, uk.company.com)
    - Country-specific subdomains (in.company.com, de.company.com, fr.company.com)
    - Departmental subdomains (hr.company.com, sales.company.com)
    
    Leave empty if no such pattern exists.
    
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

    # Build domains section
    domains_section = "\n\nTop company domains:\n"
    for i, domain in enumerate(domains[:3], 1):
        domains_section += f"{i}. {domain}\n"
        subdomains = domain_subdomains.get(domain, []) if domain_subdomains else []
        if subdomains:
            domains_section += f"   Subdomains: {', '.join(subdomains)}\n"

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

            # Check for 'domains' key first (common LLM response format)
            if "domains" in data and isinstance(data["domains"], list):
                result = data["domains"]
            else:
                # Fallback: look for any list value
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        result = value
                        break

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
        domains = ["example.com"]

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

    result = [
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

    return result


def filter_relevant_domains(
    company_name: str,
    domains: List[dict],
    llm_manager: LLMManager,
    company_info: CompanyInfo = None,
) -> List[dict]:
    """Filter domains for relevance using LLM"""
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
    
    STRICTLY REMOVE:
    - Generic domains (gmail, yahoo, hotmail, outlook)
    - Lead generation/scraping services (leadiq, zoominfo, apollo, rocketreach, hunter, clearbit, prospectstack, etc.)
    - Business directory sites (yellowpages, yelp, linkedin, crunchbase, etc.)
    - Social media platforms (facebook, twitter, instagram, etc.)
    - Unrelated companies with different business focus
    - Personal domains (individual names not matching company)
    - Job boards and recruiting sites
    - Marketing/advertising platforms
    - Any domain that provides leads/contact info about companies rather than being the company itself
    
    KEEP AND CLEAN:
    - Primary company domains (clean up if malformed)
    - Official website domain and its variations
    - Subsidiary domains (clearly related to main company)
    - Department/division domains (hr.company.com, sales.company.com)
    - Regional domains (compound TLDs like .co.uk, .edu.bd)
    - Domains that ARE the company, not domains that talk ABOUT the company
    
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
                confidence += 0.3  # LLM suggestions (company research)

            # Higher weight for email count
            if domain_data["email_count"] > 0:
                confidence += min(0.5, domain_data["email_count"] * 0.015)

                # Bonus for domains with many emails
                if domain_data["email_count"] >= 10:
                    confidence += 0.1
                elif domain_data["email_count"] >= 5:
                    confidence += 0.05

            if domain_data["source_count"] > 0:
                confidence += min(0.2, domain_data["source_count"] * 0.02)

            # Bonus for likely official domains
            if any(
                word in domain_data["domain"] for word in company_name.lower().split()
            ):
                confidence += 0.2

            domain_data["confidence"] = round(min(1.0, confidence), 3)

        return list(domain_map.values()) if domain_map else domains
    except Exception:
        return domains


def extract_email_patterns(
    domains: List[DomainResult], scraped_emails: List[str]
) -> List[str]:
    """Extract common email patterns from scraped emails"""
    patterns = set()

    for email in scraped_emails:
        if "@" in email:
            local, domain = email.split("@", 1)
            # Generate pattern from local part
            if "." in local:
                patterns.add("firstname.lastname@domain.com")
            elif len(local.split()) == 1 and len(local) > 6:
                patterns.add("firstnamelastname@domain.com")
            else:
                patterns.add("firstname@domain.com")

    # Add common patterns if none found
    if not patterns:
        patterns = {
            "firstname.lastname@domain.com",
            "firstname@domain.com",
            "f.lastname@domain.com",
            "firstnamelastname@domain.com",
        }

    return list(patterns)


# ============================================================================
# MAIN CONTACT FINDER CLASS
# ============================================================================


class ContactFinder:
    """
    Unified contact finder class that can handle both company-only and company+employee searches.

    Usage:
        finder = ContactFinder()

        # Company-only search
        result = finder.find_contacts("Acme Corp")

        # Company + employee search
        result = finder.find_contacts("Acme Corp", employee_name="John Doe")

        # With context
        result = finder.find_contacts(
            "Acme Corp",
            employee_name="John Doe",
            company_context={"industry": "Technology"},
            employee_context={"title": "CEO"}
        )
    """

    def __init__(self, serper_api_key: str = None, provider_order: List[str] = None):
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY is required")

        self.llm_manager = LLMManager(provider_order)
        self.web_scraper = WebScraper(self.serper_api_key)
        self.domain_validator = DomainValidator()

    def find_contacts(
        self,
        company_name: str,
        employee_name: Optional[str] = None,
        company_context: Dict[str, Any] = None,
        employee_context: Dict[str, Any] = None,
        max_results: int = 5,
    ) -> ContactFinderResult:
        """
        Find contact information for a company and optionally an employee.

        Args:
            company_name: Name of the company to search for
            employee_name: Name of the employee (optional)
            company_context: Additional context for company identification
            employee_context: Additional context for employee identification
            max_results: Maximum number of results to return

        Returns:
            ContactFinderResult with consistent schema
        """

        # Step 1: Research company
        company_info = research_company(company_name, self.llm_manager, company_context)

        # Step 2: Search for emails and scrape data
        search_results = self.web_scraper.search_emails(
            company_info.company_name, max_results
        )
        scraped_data = self.web_scraper.scrape_all(search_results)

        # Step 3: Analyze domains
        domain_analysis = self.domain_validator.analyze_domains(
            scraped_data, company_info.likely_email_domains, self.llm_manager
        )

        # Step 4: Filter for relevance
        domain_dicts = [d.model_dump() for d in domain_analysis]
        filtered_dicts = filter_relevant_domains(
            company_name, domain_dicts, self.llm_manager, company_info
        )

        # Step 5: Convert back to DomainResult objects and validate MX records
        filtered_results = [DomainResult(**d) for d in filtered_dicts]
        final_domain_results = []
        for result in filtered_results:
            if self.domain_validator.has_mx_record(result.domain):
                result.mx_valid = True
                final_domain_results.append(result)

        # Sort by confidence
        final_domain_results = sorted(
            final_domain_results, key=lambda x: x.confidence, reverse=True
        )

        # Step 6: Extract email patterns
        email_patterns = extract_email_patterns(
            final_domain_results, scraped_data["emails"]
        )

        # Step 7: Get ranked subdomains
        all_subdomains = []
        for domain in final_domain_results:
            all_subdomains.extend(domain.sub_mail_domains)
        # Remove duplicates while preserving order
        ranked_subdomains = list(dict.fromkeys(all_subdomains))

        # Step 8: Handle employee search if provided
        likely_emails = []
        employee_description = ""

        if employee_name:
            # Get top 3 domains for employee search
            top_domains = [result.domain for result in final_domain_results[:3]]
            domain_subdomains = {
                result.domain: result.sub_mail_domains
                for result in final_domain_results[:3]
            }

            # Search for employee-specific emails
            scraped_employee_emails = self.web_scraper.search_employee_emails(
                company_name, employee_name, top_domains, max_results=5
            )

            # Research employee emails with LLM
            email_data = research_employee_emails(
                company_name,
                employee_name,
                top_domains,
                self.llm_manager,
                employee_context,
                scraped_employee_emails,
                domain_subdomains,
            )

            # Ensure we always have results
            if not email_data:
                email_data = generate_fallback_emails(employee_name, top_domains)

            # Convert to EmailResult objects
            for email_dict in email_data:
                try:
                    email = email_dict.get("email", "")
                    if not email or "@" not in email:
                        continue

                    # Clean email
                    cleaned_email = clean_email(email)
                    if not cleaned_email:
                        continue

                    # Extract domain from email
                    email_domain = cleaned_email.split("@")[1]

                    # Create EmailResult
                    result = EmailResult(
                        email=cleaned_email,
                        confidence=email_dict.get("confidence", 0.5),
                        pattern_type=email_dict.get("pattern_type", "unknown"),
                        source=email_dict.get("source", "llm"),
                        domain=email_domain,
                        pattern_frequency=email_dict.get("pattern_frequency", 0),
                    )

                    likely_emails.append(result)

                except Exception:
                    continue

            # Sort by confidence and remove duplicates
            likely_emails.sort(key=lambda x: x.confidence, reverse=True)
            seen = set()
            unique_emails = []
            for result in likely_emails:
                if result.email not in seen:
                    seen.add(result.email)
                    unique_emails.append(result)

            likely_emails = unique_emails[:max_results]

            # Generate employee description
            if employee_context:
                desc_parts = []
                if employee_context.get("title"):
                    desc_parts.append(f"Title: {employee_context['title']}")
                if employee_context.get("department"):
                    desc_parts.append(f"Department: {employee_context['department']}")
                if employee_context.get("linkedin"):
                    desc_parts.append(f"LinkedIn: {employee_context['linkedin']}")
                employee_description = "; ".join(desc_parts)

        # Step 9: Build and return result
        return ContactFinderResult(
            company_name=company_info.company_name,
            website=company_info.website,
            likely_email_domains=[d.domain for d in final_domain_results],
            email_patterns=email_patterns,
            description=company_info.description,
            sub_mail_domains=ranked_subdomains,
            employee_name=employee_name,
            employee_description=employee_description,
            likely_email_addresses=likely_emails,
        )


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================


def find_contact_info(
    company_name: str,
    employee_name: Optional[str] = None,
    company_context: Dict[str, Any] = None,
    employee_context: Dict[str, Any] = None,
    max_results: int = 5,
) -> ContactFinderResult:
    """
    Main contact finder function - unified interface for both company and employee search.

    Args:
        company_name: Name of the company to search for
        employee_name: Name of the employee (optional)
        company_context: Additional context for company identification
        employee_context: Additional context for employee identification
        max_results: Maximum number of results to return

    Returns:
        ContactFinderResult with consistent schema
    """
    finder = ContactFinder()
    return finder.find_contacts(
        company_name=company_name,
        employee_name=employee_name,
        company_context=company_context,
        employee_context=employee_context,
        max_results=max_results,
    )
