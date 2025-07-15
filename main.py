import time
import dotenv
import threading
from domainfinder import DomainFinder


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
    # finder = DomainFinder()

    # Or specify preferred order: gpt -> gemini
    finder = DomainFinder(provider_order=["gpt", "gemini"])

    company = "Daffodil International University"
    context = {}

    print(f"🏢 Finding domains for: {company}")
    print(f"🔄 LLM Provider order: {' → '.join(finder.llm_manager.provider_order)}")
    print()

    # Start loading spinner
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=loading_spinner, args=(stop_spinner,))
    spinner_thread.start()

    # Track timing
    start_time = time.time()

    try:
        results = finder.find_domains(company, context=context)

        # Stop spinner
        stop_spinner.set()
        spinner_thread.join()

        # Clear spinner line
        print("\r" + " " * 20 + "\r", end="")

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Analysis complete in {duration:.2f}s")
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

    except Exception as e:
        # Stop spinner on error
        stop_spinner.set()
        spinner_thread.join()
        print("\r" + " " * 20 + "\r", end="")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
