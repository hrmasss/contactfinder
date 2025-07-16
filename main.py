import time
import dotenv
import threading
from domainfinder import DomainFinder
from emailfinder import EmailFinder


def loading_spinner(stop_event):
    """Show a loading spinner while processing"""
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        print(f"\r{spinner[i % len(spinner)]} Processing...", end="", flush=True)
        time.sleep(0.1)
        i += 1


def main():
    dotenv.load_dotenv()

    # Default: gemini -> gpt
    # provider_order = ["gemini", "gpt"]
    # Custom: gpt -> gemini
    provider_order = ["gpt", "gemini"]

    # Test data
    company = "ABB"
    company_context = {}
    employee = "Kevin Durocher"
    employee_context = {}

    # Initialize domain finder
    domain_finder = DomainFinder(provider_order=provider_order)

    # Test domain finder
    print("🔍 DOMAIN FINDER TEST")
    print("=" * 50)

    # Start spinner
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=loading_spinner, args=(stop_spinner,))
    spinner_thread.start()

    try:
        start_time = time.time()
        results = domain_finder.find_domains(company, context=company_context)
        stop_spinner.set()
        spinner_thread.join()

        # Clear spinner
        print("\r" + " " * 20 + "\r", end="")

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Analysis complete in {duration:.2f}s")
        print(f"📊 Found {len(results)} ranked domains:")
        print("-" * 50)

        for i, domain in enumerate(results, 1):
            subdomain_info = (
                f" (subdomains: {', '.join(domain.sub_mail_domains)})"
                if domain.sub_mail_domains
                else ""
            )
            print(
                f"{i:2d}. {domain.domain:<25} ({domain.confidence:.3f}) - {domain.email_count} emails{subdomain_info}"
            )

        if results:
            print(f"\n� Best domain: {results[0].domain}")
        else:
            print("\n⚠️ No domains found")

    except Exception as e:
        # Stop spinner on error
        stop_spinner.set()
        spinner_thread.join()
        print("\r" + " " * 20 + "\r", end="")
        print(f"❌ Domain finder error: {e}")
        import traceback

        traceback.print_exc()

    # Test email finder
    print("\n\n📧 EMAIL FINDER TEST")
    print("=" * 50)

    # Initialize email finder
    email_finder = EmailFinder(provider_order=provider_order)

    # Start spinner
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=loading_spinner, args=(stop_spinner,))
    spinner_thread.start()

    try:
        start_time = time.time()
        email_results = email_finder.find_emails(
            employee_name=employee,
            company_name=company,
            company_context=company_context,
            employee_context=employee_context,
            max_results=5,
        )
        stop_spinner.set()
        spinner_thread.join()

        # Clear spinner
        print("\r" + " " * 20 + "\r", end="")

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Analysis complete in {duration:.2f}s")
        print(f"� Found {len(email_results)} potential emails:")
        print("-" * 50)

        for i, email in enumerate(email_results, 1):
            print(
                f"{i:2d}. {email.email:<35} ({email.confidence:.3f}) - {email.pattern_type}"
            )

        if email_results:
            print(f"\n💡 Best email: {email_results[0].email}")
        else:
            print("\n⚠️ No emails found")

    except Exception as e:
        # Stop spinner on error
        stop_spinner.set()
        spinner_thread.join()
        print("\r" + " " * 20 + "\r", end="")
        print(f"❌ Email finder error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
