"""Streamlit web interface for AdvisoryAI Jarvis"""
import streamlit as st
import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import LangGraph version, fallback to simple version
JARVIS_AVAILABLE = False
JarvisAgent = None
try:
    from jarvis_graph import JarvisAgent
    JARVIS_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    error_msg = str(e)
    if "Azure OpenAI credentials" in error_msg or "AZURE_OPENAI" in error_msg:
        # Credentials missing - this will be handled in initialize_agent
        JARVIS_AVAILABLE = False
        JarvisAgent = None
    else:
        print(f"Warning: Could not import LangGraph version: {e}")
        print("Falling back to simplified version...")
        try:
            from jarvis_simple import JarvisAgentSimple as JarvisAgent
            JARVIS_AVAILABLE = True
        except ImportError:
            JARVIS_AVAILABLE = False
            print("Error: Could not import any Jarvis agent version")
from data_generator import generate_all_clients
from vector_store import ClientVectorStore
from compliance_tracker import ComplianceTracker
from config import ADVISOR_NAME, get_azure_openai_config

# Initialize Azure OpenAI config from Streamlit secrets (after Streamlit is initialized)
import config
try:
    endpoint, api_key, deployment, api_version = get_azure_openai_config()
    config.AZURE_OPENAI_ENDPOINT = endpoint
    config.AZURE_OPENAI_API_KEY = api_key
    config.AZURE_OPENAI_DEPLOYMENT = deployment
    config.AZURE_OPENAI_API_VERSION = api_version
except Exception as e:
    # If secrets not available, config will use environment variables
    # This is fine - will be checked when JarvisAgent initializes
    pass


# Page config
st.set_page_config(
    page_title="AdvisoryAI Jarvis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .priority-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .priority-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_agent():
    """Initialize and cache the Jarvis agent"""
    data_file = Path("mock_data.json")
    vector_store_path = Path("chroma_db")
    
    # Generate mock data if needed
    if not data_file.exists():
        with st.spinner("Generating mock client data..."):
            data = generate_all_clients(200)
            with open(data_file, "w") as f:
                json.dump(data, f, indent=2)
    
    # Initialize vector store
    vector_store = ClientVectorStore()
    if not vector_store_path.exists() or len(list(vector_store_path.iterdir())) == 0:
        with st.spinner("Indexing client data..."):
            vector_store.load_client_data(str(data_file))
    
    # Initialize agent
    agent = JarvisAgent()
    return agent


@st.cache_data
def get_compliance_data():
    """Get compliance tracking data"""
    tracker = ComplianceTracker()
    return tracker.get_daily_briefing()


def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<p class="main-header">🤖 AdvisoryAI Jarvis</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Proactive Assistant for {ADVISOR_NAME}</p>', unsafe_allow_html=True)
    
    # Try to initialize config from Streamlit secrets here (after Streamlit is initialized)
    global JARVIS_AVAILABLE, JarvisAgent
    try:
        endpoint, api_key, deployment, api_version = get_azure_openai_config()
        config.AZURE_OPENAI_ENDPOINT = endpoint
        config.AZURE_OPENAI_API_KEY = api_key
        config.AZURE_OPENAI_DEPLOYMENT = deployment
        config.AZURE_OPENAI_API_VERSION = api_version
        # Now try to import JarvisAgent again if it failed before
        if not JARVIS_AVAILABLE or JarvisAgent is None:
            try:
                from jarvis_graph import JarvisAgent
                JARVIS_AVAILABLE = True
            except Exception as e:
                pass
    except Exception as e:
        pass  # Will show error below
    
    # Initialize agent
    try:
        if not JARVIS_AVAILABLE or JarvisAgent is None:
            st.error("⚠️ Jarvis Agent not available")
            st.info("**Missing Azure OpenAI Credentials**")
            st.info("Please set the following environment variables in Streamlit Cloud secrets:")
            st.code("""
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
            """)
            st.info("Go to: Manage app → Settings → Secrets")
            return
        agent = initialize_agent()
    except ValueError as e:
        error_msg = str(e)
        if "Azure OpenAI credentials" in error_msg or "AZURE_OPENAI" in error_msg:
            st.error("⚠️ Missing Azure OpenAI Credentials")
            st.info("Please set the following environment variables in Streamlit Cloud secrets:")
            st.code("""
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
            """)
            st.info("Go to: Manage app → Settings → Secrets")
        else:
            st.error(f"Error initializing Jarvis: {e}")
        return
    except Exception as e:
        st.error(f"Error initializing Jarvis: {e}")
        st.info("Make sure you have set Azure OpenAI credentials in Streamlit Cloud secrets")
        if "langgraph" in str(e).lower() or "checkpoint" in str(e).lower():
            st.info("💡 Tip: The system will automatically use a simplified version if LangGraph has issues.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Quick Actions")
        
        # Use a different approach - set a query in session state instead
        if st.button("📊 Daily Briefing", use_container_width=True):
            st.session_state.pending_query = "What needs my attention today? Give me a comprehensive daily briefing."
            st.rerun()
        
        if st.button("📅 Reviews Due", use_container_width=True):
            st.session_state.pending_query = "Show me all clients due for annual review"
            st.rerun()
        
        if st.button("🎂 Milestones", use_container_width=True):
            st.session_state.pending_query = "Who has upcoming milestone birthdays?"
            st.rerun()
        
        if st.button("🎉 Life Events", use_container_width=True):
            st.session_state.pending_query = "What life events need attention?"
            st.rerun()
        
        if st.button("😟 Client Concerns", use_container_width=True):
            st.session_state.pending_query = "Show me clients with unresolved concerns"
            st.rerun()
        
        st.divider()
        
        # System Status
        st.header("System Status")
        try:
            status = agent.get_status()
            if status["fallback_mode"]:
                st.warning("⚠️ Fallback Mode Active")
                st.caption("Using direct tool execution to avoid rate limits")
            else:
                st.success("✅ Normal Mode")
                st.caption("LLM available")
            
            # Token usage indicator
            token_pct = (status["current_tokens"] / status["token_limit"]) * 100
            st.progress(min(1.0, token_pct / 100))
            st.caption(f"Tokens: {status['current_tokens']}/{status['token_limit']} ({token_pct:.1f}%)")
            
            if status["rate_limit_errors"] > 0:
                st.caption(f"⚠️ Rate limit errors: {status['rate_limit_errors']}")
            
            if status["window_reset_in"] > 0:
                st.caption(f"⏱️ Reset in: {int(status['window_reset_in'])}s")
        except Exception as e:
            st.caption(f"Status unavailable: {e}")
        
        st.divider()
        
        # Compliance summary
        st.header("Compliance Summary")
        try:
            briefing = get_compliance_data()
            summary = briefing["summary"]
            
            st.metric("Reviews Due", summary["total_reviews_due"], 
                     delta=None if summary["total_reviews_due"] == 0 else f"{summary['total_reviews_due']} urgent")
            st.metric("Contact Gaps", summary["total_contact_gaps"])
            st.metric("Overdue Actions", summary["total_overdue_actions"])
            st.metric("Milestones", summary["total_milestones"])
            st.metric("Life Events", summary["total_life_events"])
        except Exception as e:
            st.error(f"Error loading compliance data: {e}")
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat with Jarvis", "📋 Daily Briefing"])
    
    with tab1:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"Hello! I'm Jarvis, your proactive assistant. I help you stay on top of client relationships, compliance, and opportunities.\n\nTry asking me:\n• 'What needs my attention today?'\n• 'Show me reviews due'\n• 'Who has upcoming milestone birthdays?'\n• 'Tell me about clients worried about inheritance tax'"
                }
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle pending query from sidebar buttons
        if "pending_query" in st.session_state:
            user_input = st.session_state.pending_query
            del st.session_state.pending_query
        else:
            user_input = None
        
        # Chat input
        chat_input = st.chat_input("Ask Jarvis anything...")
        
        # Use chat input if provided, otherwise use pending query
        if chat_input:
            user_input = chat_input
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get response from agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = agent.chat(user_input)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_str = str(e).lower()
                        if "quota" in error_str or "429" in error_str or "insufficient_quota" in error_str:
                            error_msg = """⚠️ **API Quota Exceeded**

I'm currently unable to use the language model due to API quota limits. However, many queries work directly with your data!

**Try these queries that work without the LLM:**
- "Show me everyone with ISA allowance still available"
- "Which clients haven't had a review in over 12 months?"
- "Show me all business owners"
- "What documents am I still waiting for from clients?"
- "Show me all open action items"
- "Which clients have cash excess above 6 months expenditure?"
- "Which clients have protection gaps?"
- "Daily briefing"

These queries read directly from your data and don't require the language model."""
                            st.warning(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        else:
                            error_msg = f"Sorry, I encountered an error: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with tab2:
        st.header("Daily Briefing")
        
        try:
            briefing = get_compliance_data()
            summary = briefing["summary"]
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Reviews Due",
                    summary["total_reviews_due"],
                    delta=f"{summary['total_reviews_due']} urgent" if summary["total_reviews_due"] > 0 else None,
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Contact Gaps",
                    summary["total_contact_gaps"],
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Overdue Actions",
                    summary["total_overdue_actions"],
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "Milestones",
                    summary["total_milestones"]
                )
            
            # Reviews Due
            if briefing["reviews_due"]:
                st.subheader("🔴 Reviews Due")
                for review in briefing["reviews_due"][:10]:
                    priority_class = "priority-high" if review["priority"] == "high" else "priority-medium"
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{review['client_name']}</strong> - 
                        <span class="{priority_class}">{review['days_overdue']} days overdue</span><br>
                        Last review: {review['last_review']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
            
            # Upcoming Milestones
            if briefing["upcoming_milestones"]:
                st.subheader("🎂 Upcoming Milestones")
                for milestone in briefing["upcoming_milestones"][:10]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{milestone['client_name']}</strong> turning <strong>{milestone['turning_age']}</strong> 
                        in {milestone['days_until']} days ({milestone['birthday_date']})<br>
                        <em>Opportunity: {milestone['opportunity']}</em>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
            
            # Life Events
            if briefing["life_events"]:
                st.subheader("🎉 Life Events Requiring Attention")
                for event in briefing["life_events"][:10]:
                    days_text = f"{abs(event['days_until'])} days ago" if event['days_until'] < 0 else f"in {event['days_until']} days"
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{event['client_name']}</strong>: {event['event_type']} ({days_text})<br>
                        <em>Action: {event['action_required']}</em>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
            
            # Unresolved Concerns
            if briefing["unresolved_concerns"]:
                st.subheader("😟 Unresolved Concerns")
                for concern in briefing["unresolved_concerns"][:10]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{concern['client_name']}</strong>: {concern['concern']}<br>
                        Status: {concern['status']} | First mentioned: {concern['first_mentioned']}<br>
                        <em>Action: {concern['action_required']}</em>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
            
            # Overdue Actions
            if briefing["overdue_actions"]:
                st.subheader("⚠️ Overdue Action Items")
                for action in briefing["overdue_actions"][:10]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{action['client_name']}</strong>: {action['action']}<br>
                        Due: {action['due_date']} ({action['days_overdue']} days overdue)
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
            
            if not any([
                briefing["reviews_due"],
                briefing["upcoming_milestones"],
                briefing["life_events"],
                briefing["unresolved_concerns"],
                briefing["overdue_actions"]
            ]):
                st.success("🎉 All caught up! No urgent items requiring attention.")
        
        except Exception as e:
            st.error(f"Error loading briefing: {e}")


if __name__ == "__main__":
    main()

