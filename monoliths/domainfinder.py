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
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = set(re.findall(pattern, text, re.IGNORECASE))

        filtered = set()
        for email in emails:
            email = email.lower().strip()

            if any(
                skip in email
                for skip in ["example.", "test@", "noreply", "no-reply", "placeholder"]
            ):
                continue

            if len(email) < 5 or "@" not in email:
                continue

            try:
                local_part, domain = email.split("@", 1)
                if "." in domain and len(domain) > 3:
                    filtered.add(email)
            except ValueError:
                continue

        return filtered

    def scrape_page(self, url: str) -> Set[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=8)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                text_emails = self.extract_emails(soup.get_text())

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
                            if domain and "." in domain:
                                domain_sources.setdefault(domain, []).append(
                                    result.link
                                )
                        except Exception:
                            continue

        return {"emails": list(all_emails), "domain_sources": domain_sources}


class DomainValidator:
    def clean_domain(self, domain: str) -> str:
        if not domain:
            return ""

        domain = domain.strip().lower()

        if domain.startswith("http://") or domain.startswith("https://"):
            domain = domain.split("://", 1)[1]

        if domain.startswith("www."):
            domain = domain[4:]

        domain = domain.split("/")[0]
        domain = re.sub(r"[^a-zA-Z0-9.-]+.*$", "", domain)

        known_tlds = ["com", "org", "net", "edu", "gov", "io", "tv", "fm", "co"]

        for tld in known_tlds:
            pattern = rf"\.{tld}[a-zA-Z]+"
            if re.search(pattern, domain):
                parts = domain.split(f".{tld}")
                if len(parts) > 1:
                    domain = f"{parts[0]}.{tld}"
                    break

        if not re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", domain):
            return ""

        if len(domain) > 100:
            return ""

        return domain

    def has_mx_record(self, domain: str) -> bool:
        try:
            domain = self.clean_domain(domain)
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
        if not domains:
            return {}

        results = {}
        for domain in domains:
            if domain:
                results[domain] = self.has_mx_record(domain)

        return results

    def analyze_domains(
        self, scraped_data: Dict, llm_domains: List[str], mx_results: Dict[str, bool]
    ) -> List[DomainResult]:
        emails = scraped_data["emails"]
        domain_sources = scraped_data["domain_sources"]

        domain_counts = Counter()
        for email in emails:
            if "@" in email:
                try:
                    domain = email.split("@")[1]
                    domain = self.clean_domain(domain)
                    if domain:
                        domain_counts[domain] += 1
                except Exception:
                    continue

        clean_llm_domains = []
        for domain in llm_domains:
            clean_d = self.clean_domain(domain)
            if clean_d:
                clean_llm_domains.append(clean_d)

        all_domains = set(domain_counts.keys()) | set(clean_llm_domains)
        valid_domains = {d for d in all_domains if mx_results.get(d, False)}

        results = []
        for domain in valid_domains:
            email_count = domain_counts.get(domain, 0)
            source_count = len(domain_sources.get(domain, []))
            from_llm = domain in clean_llm_domains

            confidence = 0.0
            if from_llm:
                confidence += 0.3
            if email_count > 0:
                confidence += min(0.5, email_count * 0.05)
            if source_count > 0:
                confidence += min(0.3, source_count * 0.03)

            results.append(
                DomainResult(
                    domain=domain,
                    email_count=email_count,
                    source_count=source_count,
                    confidence=round(min(1.0, confidence), 3),
                    from_llm=from_llm,
                    mx_valid=True,
                )
            )

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results


# ============================================================================
# LLM FUNCTIONS
# ============================================================================


def research_company(company_query: str, llm_manager: LLMManager) -> CompanyInfo:
    prompt = f"""
    Research "{company_query}" and provide ONLY the JSON response below.
    
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
        self, company_query: str, max_results: int = 30
    ) -> List[DomainResult]:
        """Find and rank domains by confidence score"""

        # Research company
        company_info = research_company(company_query, self.llm_manager)

        # Search for emails
        search_results = self.web_scraper.search_emails(
            company_info.company_name, max_results
        )

        # Scrape pages
        scraped_data = self.web_scraper.scrape_all(search_results)

        # Validate domains
        all_domains = (
            list(scraped_data["domain_sources"].keys())
            + company_info.likely_email_domains
        )
        all_domains = list(set(all_domains))
        mx_results = self.domain_validator.validate_domains(all_domains)

        # Analyze domains
        domain_analysis = self.domain_validator.analyze_domains(
            scraped_data, company_info.likely_email_domains, mx_results
        )

        # Filter for relevance
        domain_dicts = [d.model_dump() for d in domain_analysis]
        filtered_dicts = filter_relevant_domains(
            company_query, domain_dicts, self.llm_manager
        )

        return [DomainResult(**d) for d in filtered_dicts]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # Initialize finder
    finder = DomainFinder()

    # Find domains
    company = "Daffodil International University"
    results = finder.find_domains(company)

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
