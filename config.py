"""Configuration settings for AdvisoryAI Jarvis"""
import os
from dotenv import load_dotenv

# Try to load .env file, but don't fail if it doesn't exist or can't be read
try:
    load_dotenv()
except Exception:
    # If .env file can't be loaded, continue with defaults
    pass

def get_azure_openai_config():
    """Get Azure OpenAI configuration from Streamlit secrets or environment variables"""
    endpoint = None
    api_key = None
    deployment = "gpt-4o-mini"
    api_version = "2024-02-15-preview"
    
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets is not None:
            secrets = st.secrets
            # Try multiple access methods
            # Method 1: Direct dictionary access with []
            try:
                if "AZURE_OPENAI_ENDPOINT" in secrets:
                    endpoint = secrets["AZURE_OPENAI_ENDPOINT"]
                if "AZURE_OPENAI_API_KEY" in secrets:
                    api_key = secrets["AZURE_OPENAI_API_KEY"]
                if "AZURE_OPENAI_DEPLOYMENT" in secrets:
                    deployment = secrets["AZURE_OPENAI_DEPLOYMENT"]
                if "AZURE_OPENAI_API_VERSION" in secrets:
                    api_version = secrets["AZURE_OPENAI_API_VERSION"]
            except (KeyError, TypeError):
                pass
            
            # Method 2: Try .get() method
            if not endpoint:
                try:
                    endpoint = secrets.get("AZURE_OPENAI_ENDPOINT")
                except (AttributeError, TypeError):
                    pass
            if not api_key:
                try:
                    api_key = secrets.get("AZURE_OPENAI_API_KEY")
                except (AttributeError, TypeError):
                    pass
            
            # Method 3: Attribute access (Streamlit secrets object)
            if not endpoint:
                try:
                    endpoint = getattr(secrets, 'AZURE_OPENAI_ENDPOINT', None)
                except (AttributeError, TypeError):
                    pass
            if not api_key:
                try:
                    api_key = getattr(secrets, 'AZURE_OPENAI_API_KEY', None)
                except (AttributeError, TypeError):
                    pass
            if deployment == "gpt-4o-mini":
                try:
                    deployment = getattr(secrets, 'AZURE_OPENAI_DEPLOYMENT', deployment)
                except (AttributeError, TypeError):
                    pass
            if api_version == "2024-02-15-preview":
                try:
                    api_version = getattr(secrets, 'AZURE_OPENAI_API_VERSION', api_version)
                except (AttributeError, TypeError):
                    pass
    except (ImportError, AttributeError, RuntimeError, Exception):
        pass
    
    # Fall back to environment variables if secrets not found
    if not endpoint:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if deployment == "gpt-4o-mini":
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", deployment)
    if api_version == "2024-02-15-preview":
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", api_version)
    
    if not endpoint or not api_key:
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
    
    return endpoint, api_key, deployment, api_version

# Get configuration (will be called after Streamlit is initialized)
# Don't initialize at import time - wait for explicit initialization
AZURE_OPENAI_ENDPOINT = None
AZURE_OPENAI_API_KEY = None
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# Try to initialize from environment variables only (for non-Streamlit contexts)
# Streamlit secrets will be loaded later in app.py
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if os.getenv("AZURE_OPENAI_DEPLOYMENT"):
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if os.getenv("AZURE_OPENAI_API_VERSION"):
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

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

