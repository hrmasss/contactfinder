import os
from typing import List, Dict, Any
from .schema import EmailResult
from .imports import DomainFinder, LLMManager
from .llm import (
    research_employee_emails,
    scrape_employee_emails,
    filter_and_rank_emails,
)


class EmailFinder:
    """
    Email finder class that finds likely email addresses for employees.

    Usage:
        finder = EmailFinder()
        results = finder.find_emails("John Doe", "Acme Corp")

        # With employee context
        context = {
            "title": "Software Engineer",
            "department": "Engineering",
            "linkedin": "linkedin.com/in/johndoe"
        }
        results = finder.find_emails("John Doe", "Acme Corp", employee_context=context)
    """

    def __init__(self, serper_api_key: str = None, provider_order: List[str] = None):
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY is required")

        self.llm_manager = LLMManager(provider_order)
        self.domain_finder = DomainFinder(self.serper_api_key, provider_order)

    def find_emails(
        self,
        employee_name: str,
        company_name: str,
        company_context: Dict[str, Any] = None,
        employee_context: Dict[str, Any] = None,
        max_results: int = 5,
    ) -> List[EmailResult]:
        """Find and rank email addresses for an employee

        Args:
            employee_name: Name of the employee
            company_name: Name of the company
            company_context: Additional context for company identification
                           e.g., {"industry": "Technology", "location": "New York", "website": "..."}
            employee_context: Additional context for employee identification
                            e.g., {"title": "CEO", "department": "Engineering", "linkedin": "..."}
            max_results: Maximum number of email results to return (1-15)

        Returns:
            List of EmailResult objects sorted by confidence
        """

        # Step 1: Get top 3 domains using domain finder
        domain_results = self.domain_finder.find_domains(
            company_name, max_results=5, context=company_context
        )

        if not domain_results:
            return []

        # Get top 3 domains and their subdomains
        top_domains = [result.domain for result in domain_results[:3]]
        domain_subdomains = {
            result.domain: result.sub_mail_domains for result in domain_results[:3]
        }

        # Step 2: Scrape for employee emails
        scraped_emails = scrape_employee_emails(
            company_name,
            employee_name,
            top_domains,
            self.serper_api_key,
            max_results=5,
        )

        # Step 3: Research with LLM
        email_data = research_employee_emails(
            company_name,
            employee_name,
            top_domains,
            self.llm_manager,
            employee_context,
            scraped_emails,
            domain_subdomains,
        )

        if not email_data:
            return []

        # Step 4: Filter and rank results
        results = filter_and_rank_emails(email_data, top_domains)

        # Limit results based on max_results
        final_results = results[:max_results]

        return final_results

    def find_emails_simple(
        self,
        employee_name: str,
        company_name: str,
        employee_context: Dict[str, Any] = None,
    ) -> List[str]:
        """Simplified interface that returns only email addresses

        Args:
            employee_name: Name of the employee
            company_name: Name of the company
            employee_context: Additional context for employee identification

        Returns:
            List of email addresses sorted by confidence
        """
        results = self.find_emails(
            employee_name, company_name, employee_context=employee_context
        )
        return [result.email for result in results]
