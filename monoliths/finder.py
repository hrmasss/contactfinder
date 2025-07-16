import os
import re
import json
import time
import requests
import dns.resolver
from google import genai
from bs4 import BeautifulSoup
from google.genai import types
from collections import Counter
from langchain_core.tools import Tool
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper


# ============================================================================
# MODELS
# ============================================================================


@dataclass
class EmailResult:
    email: str
    confidence: float
    pattern_type: str = "generated"
    domain: str = ""


@dataclass
class ContactResult:
    company_name: str
    website: str = ""
    domains: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    description: str = ""
    employee_name: Optional[str] = None
    emails: List[EmailResult] = field(default_factory=list)


# ============================================================================
# UTILITIES
# ============================================================================


def clean_domain(domain: str) -> str:
    """Clean and validate domain"""
    if not domain:
        return ""

    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = re.sub(r"^www\.", "", domain)
    domain = domain.split("/")[0].split("?")[0].split("#")[0]
    domain = re.sub(r'[,;:!?\s\[\](){}\'""].*$', "", domain)

    if not domain or "." not in domain or len(domain) < 4:
        return ""
    if domain.startswith(".") or domain.endswith(".") or ".." in domain:
        return ""

    return domain


def clean_email(email: str) -> str:
    """Clean and validate email"""
    if not email or "@" not in email:
        return ""

    email = email.strip().lower()
    if email.count("@") != 1:
        return ""

    local, domain = email.split("@")
    domain = clean_domain(domain)

    if not domain or not local:
        return ""

    return f"{local}@{domain}"


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response"""
    # Find JSON in code blocks
    if "```json" in text:
        json_part = text.split("```json")[1].split("```")[0]
    elif "```" in text and "{" in text:
        parts = text.split("```")
        json_part = next((part for part in parts if "{" in part), text)
    else:
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            json_part = text[start : end + 1]
        else:
            return {}

    try:
        result = json.loads(json_part.strip())
        return result if isinstance(result, dict) else {"domains": result}
    except json.JSONDecodeError:
        return {}


def has_mx_record(domain: str) -> bool:
    """Check if domain has MX or A records"""
    try:
        domain = clean_domain(domain)
        if not domain:
            return False

        try:
            dns.resolver.resolve(domain, "MX")
            return True
        except dns.resolver.NoAnswer:
            dns.resolver.resolve(domain, "A")
            return True
    except Exception:
        return False


# ============================================================================
# LLM PROVIDERS
# ============================================================================


class LLMProvider:
    def __init__(self):
        self.providers = []
        self._setup_providers()

    def _setup_providers(self):
        """Setup available providers"""
        if os.getenv("GOOGLE_API_KEY"):
            self.providers.append(self._gemini)
        if os.getenv("OPENAI_API_KEY"):
            self.providers.append(self._gpt)

        if not self.providers:
            raise RuntimeError("No LLM providers available")

    def _gemini(self, prompt: str) -> str:
        """Gemini with Google Search"""
        client = genai.Client()
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        return response.text

    def _gpt(self, prompt: str) -> str:
        """GPT with search capabilities"""
        llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
        search = GoogleSerperAPIWrapper()

        agent = create_react_agent(
            llm,
            [
                Tool(
                    name="search", func=search.run, description="Search for information"
                )
            ],
        )

        events = agent.stream({"messages": [("user", prompt)]}, stream_mode="values")
        for event in events:
            pass

        return event["messages"][-1].content

    def query(self, prompt: str) -> dict:
        """Query LLM providers with fallback"""
        for provider in self.providers:
            try:
                response = provider(prompt)
                data = extract_json(response)
                if data:
                    return data
            except Exception as e:
                if "503" in str(e) or "overloaded" in str(e).lower():
                    time.sleep(2)
                continue

        return {}


# ============================================================================
# WEB SCRAPER
# ============================================================================


class WebScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def search_company(self, company: str, max_results: int = 20) -> List[str]:
        """Search for company email addresses"""
        queries = [
            f'"{company}" email contact',
            f'"{company}" employees email directory',
            f'"{company}" staff contact information',
        ]

        all_emails = set()

        for query in queries:
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": max_results // len(queries)},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    for result in data.get("organic", []):
                        text = f"{result.get('title', '')} {result.get('snippet', '')}"
                        emails = re.findall(
                            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text
                        )
                        all_emails.update(emails)

                        # Scrape the actual page
                        page_emails = self._scrape_page(result.get("link", ""))
                        all_emails.update(page_emails)

            except Exception:
                continue

        # Clean and filter emails
        clean_emails = []
        for email in all_emails:
            cleaned = clean_email(email)
            if cleaned and not any(
                skip in cleaned for skip in ["example", "test", "noreply"]
            ):
                clean_emails.append(cleaned)

        return clean_emails

    def _scrape_page(self, url: str) -> List[str]:
        """Scrape emails from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=8)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text()

                # Find emails
                emails = re.findall(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text
                )

                # Add mailto links
                for link in soup.find_all("a", href=re.compile(r"^mailto:")):
                    href = link.get("href", "")
                    if href.startswith("mailto:"):
                        emails.append(href.replace("mailto:", "").split("?")[0])

                return emails
        except Exception:
            pass

        return []


# ============================================================================
# CONTACT FINDER
# ============================================================================


class ContactFinder:
    def __init__(self, serper_api_key: str = None):
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY is required")

        self.llm = LLMProvider()
        self.scraper = WebScraper(self.serper_api_key)

    def find_contacts(
        self,
        company_name: str,
        employee_name: Optional[str] = None,
        company_context: Dict[str, Any] = None,
        employee_context: Dict[str, Any] = None,
    ) -> ContactResult:
        """Find contact information for company and optionally employee"""

        # Step 1: Research company
        company_info = self._research_company(company_name, company_context)

        # Step 2: Scrape emails
        scraped_emails = self.scraper.search_company(company_name)

        # Step 3: Analyze domains with subdomain handling
        domains = self._analyze_domains(
            scraped_emails, company_info.get("likely_email_domains", []), company_name
        )

        # Step 4: Generate patterns
        patterns = self._extract_patterns(scraped_emails)

        # Step 5: Handle employee search
        employee_emails = []
        if employee_name:
            employee_emails = self._find_employee_emails(
                company_name, employee_name, domains, employee_context
            )

        return ContactResult(
            company_name=company_info.get("company_name", company_name),
            website=company_info.get("website", ""),
            domains=domains,
            patterns=patterns,
            description=company_info.get("description", ""),
            employee_name=employee_name,
            emails=employee_emails,
        )

    def _research_company(
        self, company_name: str, company_context: Dict[str, Any] = None
    ) -> dict:
        """Research company using LLM"""

        # Build context section if provided
        context_section = ""
        if company_context:
            context_section = (
                "\n\nAdditional context to identify the correct company:\n"
            )
            for key, value in company_context.items():
                if value:
                    context_section += f"- {key.title()}: {value}\n"

        prompt = f"""
        Research "{company_name}" and return company information.
        {context_section}
        
        Return JSON format:
        {{
            "company_name": "Official Company Name",
            "website": "company.com",
            "likely_email_domains": ["company.com", "subsidiary.com"],
            "description": "Brief description"
        }}
        """

        result = self.llm.query(prompt)
        return result if result else {"company_name": company_name}

    def _analyze_domains(
        self, emails: List[str], suggested_domains: List[str], company_name: str
    ) -> List[str]:
        """Analyze and rank domains using LLM for filtering and subdomain discovery"""
        domain_counts = Counter()

        # Count domains from scraped emails
        for email in emails:
            if "@" in email:
                domain = email.split("@")[1]
                domain = clean_domain(domain)
                if domain:
                    domain_counts[domain] += 1

        # Add suggested domains with higher weight
        for domain in suggested_domains:
            domain = clean_domain(domain)
            if domain:
                domain_counts[domain] += 5

        # Get top domains for LLM analysis
        top_domains = [domain for domain, count in domain_counts.most_common(20)]

        # Use LLM to filter relevant domains and find subdomains
        relevant_domains = self._llm_filter_and_discover_domains(
            company_name, top_domains
        )

        # Validate domains with MX records
        valid_domains = []
        for domain in relevant_domains:
            if has_mx_record(domain):
                valid_domains.append(domain)

        return valid_domains[:10]

    def _extract_patterns(self, emails: List[str]) -> List[str]:
        """Extract common email patterns using LLM research"""

        prompt = """
        Research common email patterns used by business organizations.
        
        Task: Identify 3-5 most common corporate email patterns used by companies.
        
        Return JSON array of patterns (use generic terms):
        ["firstname.lastname@domain.com", "firstname@domain.com", "f.lastname@domain.com"]
        """

        result = self.llm.query(prompt)

        if result and isinstance(result, dict):
            # Handle different response formats
            if "patterns" in result:
                return result["patterns"][:5]  # Max 5 patterns
            elif isinstance(result, list):
                return result[:5]
            else:
                # Extract any list value from the response
                for value in result.values():
                    if isinstance(value, list):
                        return value[:5]

        # Fallback patterns if LLM fails
        return [
            "firstname.lastname@domain.com",
            "firstname@domain.com",
            "f.lastname@domain.com",
        ]

    def _find_employee_emails(
        self,
        company_name: str,
        employee_name: str,
        domains: List[str],
        employee_context: Dict[str, Any] = None,
    ) -> List[EmailResult]:
        """Find/generate employee emails"""

        # Try to find existing emails first
        existing_emails = self._search_employee_emails(
            company_name, employee_name, domains, employee_context
        )

        if existing_emails:
            return [
                EmailResult(email=email, confidence=0.8, pattern_type="found")
                for email in existing_emails[:3]
            ]

        # Generate email patterns
        return self._generate_employee_emails(employee_name, domains)

    def _search_employee_emails(
        self,
        company_name: str,
        employee_name: str,
        domains: List[str],
        employee_context: Dict[str, Any] = None,
    ) -> List[str]:
        """Search for existing employee emails"""

        # Build context-aware queries
        base_queries = [
            f'"{employee_name}" "{company_name}" email',
            f'"{employee_name}" site:linkedin.com email',
        ]

        # Add context-specific queries
        if employee_context:
            if employee_context.get("title"):
                base_queries.append(
                    f'"{employee_name}" "{employee_context["title"]}" "{company_name}" email'
                )
            if employee_context.get("department"):
                base_queries.append(
                    f'"{employee_name}" "{employee_context["department"]}" "{company_name}" email'
                )
            if employee_context.get("location"):
                base_queries.append(
                    f'"{employee_name}" "{employee_context["location"]}" "{company_name}" email'
                )

        found_emails = set()

        for query in base_queries:
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.serper_api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": 5},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    for result in data.get("organic", []):
                        text = f"{result.get('title', '')} {result.get('snippet', '')}"
                        emails = re.findall(
                            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text
                        )

                        for email in emails:
                            cleaned = clean_email(email)
                            if cleaned and any(domain in cleaned for domain in domains):
                                found_emails.add(cleaned)

            except Exception:
                continue

        return list(found_emails)

    def _generate_employee_emails(
        self, employee_name: str, domains: List[str]
    ) -> List[EmailResult]:
        """Generate email patterns for employee using discovered patterns"""
        if not domains:
            return []

        name_parts = employee_name.lower().split()
        if len(name_parts) < 2:
            return []

        first, last = name_parts[0], name_parts[-1]

        # Use LLM to generate patterns based on employee name and domains
        prompt = f"""
        Employee: "{employee_name}"
        Company domains: {domains[:2]}
        
        Generate 10-15 most likely email addresses for this employee.
        Use common business email patterns like:
        - firstname.lastname
        - firstname
        - f.lastname
        - firstnamelastname
        - firstname_lastname
        
        Return JSON array of email objects:
        [
          {{"email": "john.doe@company.com", "confidence": 0.8}},
          {{"email": "john@company.com", "confidence": 0.6}}
        ]
        """

        result = self.llm.query(prompt)

        if result and isinstance(result, dict):
            # Handle different response formats
            emails_data = None
            if "emails" in result:
                emails_data = result["emails"]
            elif isinstance(result, list):
                emails_data = result
            else:
                # Extract any list value from the response
                for value in result.values():
                    if isinstance(value, list):
                        emails_data = value
                        break

            if emails_data:
                results = []
                for item in emails_data[:15]:  # Max 15 emails
                    if isinstance(item, dict):
                        email = item.get("email", "")
                        confidence = item.get("confidence", 0.5)
                    else:
                        email = str(item)
                        confidence = 0.5

                    if email and "@" in email:
                        domain = email.split("@")[1]
                        results.append(
                            EmailResult(
                                email=email,
                                confidence=confidence,
                                pattern_type="generated",
                                domain=domain,
                            )
                        )

                if results:
                    return results

        # Fallback to simple patterns if LLM fails
        simple_patterns = [
            f"{first}.{last}",
            f"{first}",
            f"{last}",
            f"{first}{last}",
            f"{first[0]}.{last}",
        ]

        results = []
        for domain in domains[:2]:  # Use top 2 domains
            for i, pattern in enumerate(simple_patterns):
                email = f"{pattern}@{domain}"
                confidence = 0.5 - (
                    i * 0.05
                )  # Decrease confidence for less common patterns

                results.append(
                    EmailResult(
                        email=email,
                        confidence=confidence,
                        pattern_type="fallback",
                        domain=domain,
                    )
                )

        return results[:15]  # Return top 15 patterns

    def _llm_filter_and_discover_domains(
        self, company_name: str, domains: List[str]
    ) -> List[str]:
        """Use LLM to filter relevant domains and discover subdomains"""

        domains_list = "\n".join([f"- {domain}" for domain in domains])

        prompt = f"""
        Company: "{company_name}"
        
        Found domains from scraped emails (WARNING: may contain garbage data from data brokers, social media, etc.):
        {domains_list}
        
        Task: Filter and identify EMAIL CONTACT DOMAINS only.
        
        KEEP ONLY domains that:
        - Are used for official business email communication by "{company_name}"
        - Belong to "{company_name}" or its subsidiaries/divisions
        - Have evidence of being used for employee email addresses
        
        EXCLUDE domains that are:
        - Data brokers, lead generation tools, contact databases
        - Social media platforms (LinkedIn, Twitter, etc.)
        - Personal email providers (Gmail, Outlook, Yahoo, etc.)
        - Marketing/sales tools and platforms
        - Directory services and listing sites
        
        Only suggest subdomains if you find actual evidence they are used for email communication.
        
        Return JSON array of EMAIL CONTACT DOMAINS with evidence of email usage:
        ["company.com", "subsidiary.com"]
        """

        result = self.llm.query(prompt)

        if result and isinstance(result, dict):
            # Handle different response formats
            if "domains" in result:
                return result["domains"]
            elif isinstance(result, list):
                return result
            else:
                # Extract any list value from the response
                for value in result.values():
                    if isinstance(value, list):
                        return value

        # Fallback: return original domains if LLM fails
        return domains[:10]


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================


def find_contact_info(
    company_name: str, company_context: Dict[str, Any] = None, max_results: int = 5
) -> List[str]:
    """Legacy function for backward compatibility"""
    finder = ContactFinder()
    result = finder.find_contacts(company_name, company_context=company_context)
    return result.domains[:max_results]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # Test configuration
    company = "ABB"
    employee = "Kevin Durocher"

    print("🧪 CONTACT FINDER TEST")
    print("=" * 50)

    try:
        finder = ContactFinder()

        # Test company + employee search
        if employee:
            result = finder.find_contacts(company, employee)

            print(f"✅ Company: {result.company_name}")
            print(f"🌐 Website: {result.website}")
            print(f"📊 Description: {result.description}")
            print(f"🔍 Domains ({len(result.domains)}):")
            for i, domain in enumerate(result.domains, 1):
                print(f"  {i}. {domain}")

            print(f"📧 Patterns ({len(result.patterns)}):")
            for i, pattern in enumerate(result.patterns, 1):
                print(f"  {i}. {pattern}")

            print(f"👤 Employee: {result.employee_name}")
            print(f"📬 Emails ({len(result.emails)}):")
            for i, email in enumerate(result.emails[:5], 1):
                print(f"  {i}. {email.email} ({email.confidence:.3f})")

        else:
            result = finder.find_contacts(company)
            print(f"✅ Company: {result.company_name}")
            print(f"🔍 Domains: {result.domains}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
