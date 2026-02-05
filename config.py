"""Configuration settings for AdvisoryAI Jarvis"""
import os
from dotenv import load_dotenv

# Try to load .env file, but don't fail if it doesn't exist or can't be read
try:
    load_dotenv()
except Exception:
    # If .env file can't be loaded, continue with defaults
    pass

# Azure OpenAI Configuration
# All credentials must be set via environment variables for security
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError(
        "Azure OpenAI credentials not found. Please set the following environment variables:\n"
        "- AZURE_OPENAI_ENDPOINT\n"
        "- AZURE_OPENAI_API_KEY\n"
        "Optional:\n"
        "- AZURE_OPENAI_DEPLOYMENT (default: gpt-4o-mini)\n"
        "- AZURE_OPENAI_API_VERSION (default: 2024-02-15-preview)"
    )

# Vector Store Configuration
VECTOR_STORE_PATH = "./chroma_db"
COLLECTION_NAME = "advisor_clients"

# Advisor Configuration
ADVISOR_NAME = "Sarah Mitchell"
TOTAL_CLIENTS = 200

# Compliance Settings
ANNUAL_REVIEW_PERIOD_DAYS = 365
CONTACT_GAP_WARNING_DAYS = 90
MILESTONE_BIRTHDAY_AGES = [50, 55, 60, 65, 70, 75]

# Proactive Alert Settings
DAYS_AHEAD_FOR_ALERTS = 30  # Alert on events within next 30 days

