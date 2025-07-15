import re
import time
import json
import requests
import dns.resolver
from typing import List, Dict, Set
from collections import Counter
from bs4 import BeautifulSoup
from .schema import LLMResponse, SearchResult, DomainResult


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


class LLMManager:
    def __init__(self, provider_order: List[str] = None):
        self.provider_order = provider_order or ["gemini", "gpt"]
        self.providers = {}
        self._setup_providers()

    def _setup_providers(self):
        try:
            from .agents import gemini

            self.providers["gemini"] = gemini
        except ImportError:
            pass

        try:
            from .agents import gpt

            self.providers["gpt"] = gpt
        except ImportError:
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

            # Improved confidence scoring that gives more weight to email count
            confidence = 0.0

            # Base score for LLM suggestions (company research)
            if from_llm:
                confidence += 0.3

            # Email count bonus
            if email_count > 0:
                # More generous scoring for email count
                confidence += min(0.4, email_count * 0.01)

                # Bonus for domains with many emails
                if email_count >= 10:
                    confidence += 0.15
                elif email_count >= 5:
                    confidence += 0.1

            # Source diversity bonus
            if source_count > 0:
                confidence += min(0.2, source_count * 0.02)

            # Domain name relevance bonus
            confidence += 0.1  # Base relevance score

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
