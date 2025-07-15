import os
import re
import time
import json
import requests
import dns.resolver
from bs4 import BeautifulSoup
from collections import Counter
from typing import List, Dict, Set, Any
from pydantic import BaseModel, Field, field_validator

# LLM Imports
from google import genai
from google.genai import types
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper


# ============================================================================
# COMMON UTILITIES
# ============================================================================


def _clean_domain_text(domain: str) -> str:
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


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class CompanyInfo(BaseModel):
    company_name: str
    website: str = ""
    likely_email_domains: List[str] = Field(default_factory=list)
    description: str = ""

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


class SearchResult(BaseModel):
    title: str = ""
    link: str
    snippet: str = ""


class LLMResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any] | None = None
    error: str | None = None
    provider: str


# ============================================================================
# LLM AGENTS
# ============================================================================


def gemini(prompt: str) -> str:
    """Gemini 2.0 Flash with Google Search."""
    client = genai.Client()
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config,
    )
    return response.text


def gpt(prompt: str, model: str = "gpt-4o-mini") -> str:
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
                cleaned_domain = _clean_domain_text(domain)

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
                soup = BeautifulSoup(response.content, "html.parser")

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
                            cleaned_domain = _clean_domain_text(domain)

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
            domain = _clean_domain_text(domain)
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

    def validate_domains(self, domains: List[str]) -> Dict[str, bool]:
        """Validate multiple domains for MX/A records"""
        if not domains:
            return {}

        return {domain: self.has_mx_record(domain) for domain in domains if domain}

    def analyze_domains(
        self, scraped_data: Dict, llm_domains: List[str], mx_results: Dict[str, bool]
    ) -> List[DomainResult]:
        """Analyze and rank domains by confidence score"""
        emails = scraped_data["emails"]
        domain_sources = scraped_data["domain_sources"]

        # Count email occurrences per domain
        domain_counts = Counter()
        for email in emails:
            if "@" in email:
                try:
                    domain = email.split("@")[1]
                    domain = _clean_domain_text(domain)
                    if domain:
                        domain_counts[domain] += 1
                except Exception:
                    continue

        # Add LLM-suggested domains
        for domain in llm_domains:
            domain = _clean_domain_text(domain)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0)

        # Calculate confidence scores for all domains
        results = []
        for domain, email_count in domain_counts.items():
            source_count = len(domain_sources.get(domain, []))
            from_llm = domain in llm_domains

            confidence = 0.0

            # Base score for LLM suggestions (company research)
            if from_llm:
                confidence += 0.4

            # Email count bonus (but not overwhelming)
            if email_count > 0:
                confidence += min(0.3, email_count * 0.03)

            # Source diversity bonus
            if source_count > 0:
                confidence += min(0.2, source_count * 0.02)

            # Domain name relevance bonus
            confidence += 0.1

            results.append(
                DomainResult(
                    domain=domain,
                    email_count=email_count,
                    source_count=source_count,
                    confidence=round(min(1.0, confidence), 3),
                    from_llm=from_llm,
                    mx_valid=False,
                )
            )

        return sorted(results, key=lambda x: x.confidence, reverse=True)


# ============================================================================
# LLM FUNCTIONS
# ============================================================================


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
        return domains


# ============================================================================
# MAIN DOMAIN FINDER CLASS
# ============================================================================


class DomainFinder:
    """
    Main domain finder class that orchestrates the entire process.

    Usage:
        finder = DomainFinder()
        results = finder.find_domains("Company Name")

        # With custom provider order
        finder = DomainFinder(provider_order=["gpt", "gemini"])
        results = finder.find_domains("Company Name")
    """

    def __init__(self, serper_api_key: str = None, provider_order: List[str] = None):
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY is required")

        self.llm_manager = LLMManager(provider_order)
        self.web_scraper = WebScraper(self.serper_api_key)
        self.domain_validator = DomainValidator()

    def find_domains(
        self, company_query: str, max_results: int = 30, context: Dict[str, Any] = None
    ) -> List[DomainResult]:
        """Find and rank domains by confidence score

        Args:
            company_query: Name of the company to search for
            max_results: Maximum number of search results to process
            context: Additional context to help identify the correct company
                    e.g., {"industry": "Technology", "location": "New York", "linkedin": "...", "website": "..."}
        """

        # Research company
        company_info = research_company(company_query, self.llm_manager, context)

        # Search for emails
        search_results = self.web_scraper.search_emails(
            company_info.company_name, max_results
        )

        # Scrape pages
        scraped_data = self.web_scraper.scrape_all(search_results)

        # Analyze domains (without MX filtering yet)
        domain_analysis = self.domain_validator.analyze_domains(
            scraped_data, company_info.likely_email_domains, {}
        )

        # Filter for relevance with company research context (includes cleaning)
        domain_dicts = [d.model_dump() for d in domain_analysis]
        filtered_dicts = filter_relevant_domains(
            company_query, domain_dicts, self.llm_manager, company_info
        )

        # Convert back to DomainResult objects
        filtered_results = [DomainResult(**d) for d in filtered_dicts]

        # Final MX lookup validation - remove domains without valid MX records
        final_results = []
        for result in filtered_results:
            if self.domain_validator.has_mx_record(result.domain):
                result.mx_valid = True
                final_results.append(result)

        # Re-sort by confidence after MX filtering
        return sorted(final_results, key=lambda x: x.confidence, reverse=True)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # Initialize finder
    finder = DomainFinder()

    # Find domains with context
    company = "Daffodil International University"
    context = {
        "industry": "Education",
        "location": "Bangladesh",
        "website": "diu.edu.bd",
    }

    results = finder.find_domains(company, context=context)

    print(f"🏢 {company}")
    print(f"📊 Found {len(results)} ranked domains:")
    print("-" * 50)

    for i, domain in enumerate(results, 1):
        print(
            f"{i:2d}. {domain.domain:<25} ({domain.confidence:.3f}) - {domain.email_count} emails"
        )

    if results:
        print(f"\n💡 Best domain: {results[0].domain}")
    else:
        print("\n⚠️ No domains found")
