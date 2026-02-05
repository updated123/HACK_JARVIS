# HACK_JARVIS

## AdvisoryAI Jarvis - Proactive Chatbot for Financial Advisors

A proactive AI assistant built for UK Independent Financial Advisors to help them break free from reactive work and focus on what matters: building client relationships and delivering value.

## 🎯 Problem Statement

UK Financial Advisors manage 150-250 client relationships while drowning in administrative tasks. They spend 60-70% of their time on admin instead of advice. With so many clients, they can't remember everything:
- Life events mentioned in passing
- Concerns expressed months ago
- Opportunities with time windows (milestone birthdays, deadlines)
- Follow-up commitments

Under FCA Consumer Duty regulations, advisors must demonstrate ongoing value, but tracking this manually is error-prone and creates constant anxiety.

## ✨ Solution: Jarvis

Jarvis is a proactive chatbot that:
1. **Provides daily briefings** on what needs attention (reviews due, contact gaps, milestones, life events)
2. **Remembers everything** about clients using vector search across meeting notes, profiles, and documents
3. **Surfaces opportunities** at the right moment (milestone birthdays, life events, tax planning windows)
4. **Tracks compliance** automatically (annual reviews, Consumer Duty requirements)
5. **Answers questions** about any client instantly using natural language

## 🛠️ Tech Stack

- **LangGraph**: Workflow orchestration for the chatbot agent
- **LangChain**: LLM interactions and tool integration
- **Azure OpenAI**: GPT-4o-mini for natural language understanding
- **ChromaDB**: Vector store for semantic search across client information
- **Streamlit**: Modern web interface for advisors
- **Python 3.9+**: Core programming language
- **Faker**: Mock data generation for realistic UK client profiles

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Azure OpenAI account with API credentials

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/updated123/HACK_JARVIS.git
cd HACK_JARVIS
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here

# Optional Azure OpenAI Configuration
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**Important:** Never commit your `.env` file to version control. It's already included in `.gitignore`.

#### 4. Generate Mock Data

```bash
python data_generator.py
```

This creates `mock_data.json` with 200 realistic client profiles, meeting notes, life events, and compliance data.

#### 5. Run the Application

**Streamlit Web Interface (Recommended):**

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

**CLI Interface:**

```bash
python main.py
```

## 📋 Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Yes | Your Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Yes | Your Azure OpenAI API key | `your-api-key-here` |
| `AZURE_OPENAI_DEPLOYMENT` | No | Deployment name (default: gpt-4o-mini) | `gpt-4o-mini` |
| `AZURE_OPENAI_API_VERSION` | No | API version (default: 2024-02-15-preview) | `2024-02-15-preview` |

## 🌐 Deployment

### Deploying to Streamlit Cloud (Free)

1. Push your code to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository: `updated123/HACK_JARVIS`
6. Set main file: `app.py`
7. Go to "Advanced settings" → "Secrets"
8. Add your Azure OpenAI credentials as secrets
9. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## 📁 Project Structure

```
HACK_JARVIS/
├── app.py                      # Streamlit web interface
├── main.py                     # CLI interface
├── jarvis_graph.py             # LangGraph workflow for chatbot agent
├── vector_store.py             # Vector store management
├── compliance_tracker.py       # Compliance tracking logic
├── data_generator.py           # Mock data generator
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── .streamlit/                 # Streamlit configuration
    └── config.toml
```

## 🎯 Features

- **Daily Briefing**: Comprehensive overview of reviews due, contact gaps, milestones, life events
- **Client Search**: Natural language queries about any client
- **Compliance Tracking**: Automatic tracking of annual reviews and Consumer Duty requirements
- **Proactive Insights**: Milestone birthday alerts, life event opportunities, tax planning windows

## 🔒 Security

- All credentials stored in environment variables
- `.env` file is in `.gitignore`
- Never commit API keys or secrets

## 📄 License

MIT License - Built for AdvisoryAI Hack-to-Hire Challenge

---

**Built with ❤️ for UK Financial Advisors**
