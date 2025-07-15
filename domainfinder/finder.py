import os
from typing import List, Dict, Any
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

        # Filter for relevance with company research context
        domain_dicts = [d.model_dump() for d in domain_analysis]
        filtered_dicts = filter_relevant_domains(
            company_query, domain_dicts, self.llm_manager, company_info
        )

        # Convert back to DomainResult objects
        filtered_results = [DomainResult(**d) for d in filtered_dicts]

        # Final MX lookup validation
        final_results = []
        for result in filtered_results:
            if self.domain_validator.has_mx_record(result.domain):
                result.mx_valid = True
                final_results.append(result)

        # Re-sort by confidence after MX filtering
        return sorted(final_results, key=lambda x: x.confidence, reverse=True)
