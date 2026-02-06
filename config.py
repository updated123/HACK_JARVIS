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
    """Get Azure OpenAI configuration - tries multiple methods"""
    endpoint = None
    api_key = None
    deployment = "gpt-4o-mini"
    api_version = "2024-02-15-preview"
    
    # Method 1: Try environment variables first (works everywhere)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if os.getenv("AZURE_OPENAI_API_VERSION"):
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Method 2: Try Streamlit secrets (for Streamlit Cloud)
    if not endpoint or not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and st.secrets:
                secrets = st.secrets
                # Try all access patterns
                patterns = [
                    lambda: secrets.get("AZURE_OPENAI_ENDPOINT") if hasattr(secrets, 'get') else None,
                    lambda: secrets["AZURE_OPENAI_ENDPOINT"] if "AZURE_OPENAI_ENDPOINT" in secrets else None,
                    lambda: getattr(secrets, 'AZURE_OPENAI_ENDPOINT', None),
                ]
                for pattern in patterns:
                    try:
                        if not endpoint:
                            endpoint = pattern()
                        if endpoint:
                            break
                    except:
                        continue
                
                patterns_key = [
                    lambda: secrets.get("AZURE_OPENAI_API_KEY") if hasattr(secrets, 'get') else None,
                    lambda: secrets["AZURE_OPENAI_API_KEY"] if "AZURE_OPENAI_API_KEY" in secrets else None,
                    lambda: getattr(secrets, 'AZURE_OPENAI_API_KEY', None),
                ]
                for pattern in patterns_key:
                    try:
                        if not api_key:
                            api_key = pattern()
                        if api_key:
                            break
                    except:
                        continue
        except Exception:
            pass
    
    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI credentials not found.\n\n"
            "**For Streamlit Cloud:**\n"
            "1. Go to: Manage app → Settings → Secrets\n"
            "2. Add these as environment variables (not TOML):\n"
            "   - Name: AZURE_OPENAI_ENDPOINT, Value: https://your-resource.openai.azure.com/\n"
            "   - Name: AZURE_OPENAI_API_KEY, Value: your-api-key\n"
            "3. Or use TOML format in Secrets:\n"
            "   AZURE_OPENAI_ENDPOINT = \"https://your-resource.openai.azure.com/\"\n"
            "   AZURE_OPENAI_API_KEY = \"your-api-key\"\n\n"
            "**For local development:**\n"
            "Set environment variables or use .env file"
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

