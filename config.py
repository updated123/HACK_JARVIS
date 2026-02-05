"""Configuration settings for AdvisoryAI Jarvis"""
import os
from dotenv import load_dotenv

# Try to load .env file, but don't fail if it doesn't exist or can't be read
try:
    load_dotenv()
except Exception:
    # If .env file can't be loaded, continue with defaults
    pass

# Try to get secrets from Streamlit Cloud first, then fall back to environment variables
AZURE_OPENAI_ENDPOINT = None
AZURE_OPENAI_API_KEY = None
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# Try Streamlit secrets first (for Streamlit Cloud)
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        # Access secrets directly (Streamlit Cloud format)
        try:
            AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
            AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
        except KeyError:
            pass
        try:
            AZURE_OPENAI_DEPLOYMENT = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
        except KeyError:
            pass
        try:
            AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
        except KeyError:
            pass
except (ImportError, AttributeError, RuntimeError):
    pass

# Fall back to environment variables if secrets not found
if not AZURE_OPENAI_ENDPOINT:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
if not AZURE_OPENAI_API_KEY:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if AZURE_OPENAI_DEPLOYMENT == "gpt-4o-mini":  # Only use default if not set
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
if AZURE_OPENAI_API_VERSION == "2024-02-15-preview":  # Only use default if not set
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError(
        "Azure OpenAI credentials not found. Please set the following:\n"
        "For Streamlit Cloud: Add secrets in app settings\n"
        "For local development: Set environment variables:\n"
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

