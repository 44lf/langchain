import os
from dotenv import load_dotenv

load_dotenv()

# Check required environment variables
required_vars = ['DEEPSEEK_API_KEY', 'DEEPSEEK_base_URL']
missing_vars = []

for var in required_vars:
    value = os.getenv(var)
    if not value:
        missing_vars.append(var)
        print(f"❌ {var} is not set")
    else:
        print(f"✅ {var} is set")

if missing_vars:
    print(f"\n⚠️  Missing {len(missing_vars)} environment variable(s): {', '.join(missing_vars)}")
    print("Please set them in your .env file")
else:
    print("\n✅ All required environment variables are set!")

