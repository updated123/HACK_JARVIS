# AdvisoryAI Jarvis - Proactive Chatbot for Financial Advisors

A proactive AI assistant built for UK Independent Financial Advisors to help them break free from reactive work and focus on what matters: building client relationships and delivering value.

## 📋 Project Information

**Project Name:** AdvisoryAI Jarvis

**Problem Statement:** UK Financial Advisors manage 150-250 client relationships while drowning in administrative tasks. They spend 60-70% of their time on admin instead of advice. With so many clients, they can't remember everything:
- Life events mentioned in passing
- Concerns expressed months ago
- Opportunities with time windows (milestone birthdays, deadlines)
- Follow-up commitments

Under FCA Consumer Duty regulations, advisors must demonstrate ongoing value, but tracking this manually is error-prone and creates constant anxiety.

**Solution Overview:** Jarvis is a proactive chatbot that:
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

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Azure OpenAI account with API credentials

### Step-by-Step Setup Guide

#### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd HACK
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# Copy the example file (if available) or create manually
touch .env
```

Add the following environment variables to `.env`:

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

The application uses mock data to demonstrate functionality. Generate it with:

```bash
python data_generator.py
```

This creates `mock_data.json` with 200 realistic client profiles, meeting notes, life events, and compliance data.

#### 5. Run the Application

**Option A: Streamlit Web Interface (Recommended)**

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

**Option B: CLI Interface**

```bash
python main.py
```

**Option C: Test Compliance Tracker Only**

```bash
python demo.py
```

## 🌐 Deployment

### Deploying to Streamlit Cloud (Free)

Streamlit Cloud offers free hosting for Streamlit applications.

#### Steps:

1. **Push your code to GitHub**
   - Ensure all credentials are removed from code
   - Make sure `.env` is in `.gitignore`
   - Push to a public GitHub repository

2. **Sign up for Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Add secrets (environment variables):
     - Go to "Advanced settings" → "Secrets"
     - Add your Azure OpenAI credentials:
     ```
     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
     AZURE_OPENAI_API_KEY=your-api-key-here
     AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
     AZURE_OPENAI_API_VERSION=2024-02-15-preview
     ```

4. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete
   - Your app will be live at `https://your-app-name.streamlit.app`

### Alternative Deployment Options

- **Railway**: Free tier available, supports Python apps
- **Render**: Free tier available, easy deployment
- **Heroku**: Free tier discontinued, but paid options available
- **AWS/GCP/Azure**: More complex setup, but scalable

## 📋 Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Yes | Your Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Yes | Your Azure OpenAI API key | `your-api-key-here` |
| `AZURE_OPENAI_DEPLOYMENT` | No | Deployment name (default: gpt-4o-mini) | `gpt-4o-mini` |
| `AZURE_OPENAI_API_VERSION` | No | API version (default: 2024-02-15-preview) | `2024-02-15-preview` |

## 📁 Project Structure

```
HACK/
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
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── chroma_db/                  # Vector database (generated)
└── mock_data.json              # Mock client data (generated)
```

## 🎯 Features

### Daily Briefing
Get a comprehensive overview of:
- Reviews due (with days overdue)
- Contact gaps (clients not contacted recently)
- Upcoming milestone birthdays
- Life events requiring attention
- Unresolved client concerns
- Overdue action items and follow-ups

### Client Search
Ask natural language questions:
- "Tell me about clients worried about inheritance tax"
- "Who mentioned their daughter's wedding?"
- "Show me clients with upcoming retirement"
- "What did Mrs. Patterson say in our last meeting?"

### Compliance Tracking
- Automatic tracking of annual review deadlines
- Contact gap monitoring (FCA Consumer Duty)
- Action item tracking
- Follow-up commitment reminders

### Proactive Insights
- Milestone birthday alerts (50, 55, 60, 65, 70, 75)
- Life event opportunities
- Tax planning windows
- Portfolio review triggers

## 🔒 Security Notes

- **Never commit credentials**: All API keys and secrets are stored in environment variables
- **Use `.env` file locally**: Create a `.env` file for local development (already in `.gitignore`)
- **Use platform secrets for deployment**: Use your hosting platform's secrets management for production
- **Rotate credentials**: Regularly rotate API keys for security

## 🧪 Testing

The application includes mock data for testing. To test specific components:

```bash
# Test compliance tracker
python demo.py

# Test direct queries
python direct_query.py

# Test tools
python test_tools.py
```

## 📊 Mock Data

The mock data generator creates realistic scenarios:
- 200 clients with varying ages, risk profiles, and portfolio values
- Meeting notes with transcripts and action items
- Life events (weddings, retirements, inheritances)
- Client concerns (inheritance tax, market volatility, etc.)
- Compliance deadlines (some overdue, some upcoming)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.9+)

2. **Azure OpenAI Connection Issues**
   - Verify your API key and endpoint are correct
   - Check that your Azure OpenAI resource is active
   - Ensure the deployment name matches your Azure resource

3. **Vector Store Issues**
   - Delete `chroma_db/` folder and regenerate data
   - Run `python data_generator.py` again

4. **Rate Limit Errors**
   - The app includes fallback mechanisms for rate limits
   - Check your Azure OpenAI quota limits
   - Some queries work directly without LLM (see fallback mode)

## 📝 Notes

This is a hackathon project built to demonstrate the concept. In production, it would integrate with:
- CRM systems (Intelliflo, etc.)
- Portfolio platforms
- Provider APIs
- Calendar systems
- Document management systems

## 🤝 Contributing

This is a hackathon submission. For questions or feedback, please reach out!

## 📄 License

MIT License - Built for AdvisoryAI Hack-to-Hire Challenge

---

**Built with ❤️ for UK Financial Advisors**
