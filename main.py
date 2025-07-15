import dotenv
from domainfinder import DomainFinder

dotenv.load_dotenv()
finder = DomainFinder()
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
