import os
from typing import List
from .schema import DomainResult
from .utils import LLMManager, WebScraper, DomainValidator
from .llm import research_company, filter_relevant_domains


class DomainFinder:
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

        # Convert back to DomainResult objects
        return [DomainResult(**d) for d in filtered_dicts]
