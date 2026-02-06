"""FastAPI backend for Jarvis - Deploy on Render"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Initialize FastAPI app
app = FastAPI(title="Jarvis API", version="1.0.0")

# CORS middleware - allow Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jarvis Agent
jarvis_agent = None
compliance_tracker = None
init_error_message = None

def initialize_backend():
    """Initialize backend services"""
    global jarvis_agent, compliance_tracker, init_error_message
    init_error_message = None
    
    try:
        # Step 1: Set credentials FIRST (before any imports that use config)
        import config
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not endpoint or not api_key:
            error_msg = "Missing Azure OpenAI credentials. "
            if not endpoint:
                error_msg += "AZURE_OPENAI_ENDPOINT not set. "
            if not api_key:
                error_msg += "AZURE_OPENAI_API_KEY not set."
            raise ValueError(error_msg)
        
        # Set config values
        config.AZURE_OPENAI_ENDPOINT = endpoint
        config.AZURE_OPENAI_API_KEY = api_key
        config.AZURE_OPENAI_DEPLOYMENT = deployment
        config.AZURE_OPENAI_API_VERSION = api_version
        print(f"✓ Credentials loaded (endpoint: {endpoint[:30]}..., deployment: {deployment})")
        
        # Step 2: Generate mock data if needed
        from data_generator import generate_all_clients
        import json
        data_file = Path("mock_data.json")
        if not data_file.exists():
            print("Generating mock data...")
            data = generate_all_clients(200)
            with open(data_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Generated {data.get('total_clients', 0)} clients")
        else:
            print("✓ Mock data file exists")
        
        # Step 3: Initialize vector store and load data
        from vector_store import ClientVectorStore
        vector_store_path = Path("chroma_db")
        print("Initializing vector store...")
        vector_store = ClientVectorStore()
        
        # Check if vector store needs to be populated
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            print("Vector store empty, loading client data...")
            vector_store.load_client_data(str(data_file))
            print("✓ Vector store populated")
        else:
            print("✓ Vector store already exists")
        
        # Step 4: Initialize compliance tracker
        from compliance_tracker import ComplianceTracker
        compliance_tracker = ComplianceTracker()
        print("✓ Compliance tracker initialized")
        
        # Step 5: Initialize JarvisAgent (now that everything is ready)
        from jarvis_graph import JarvisAgent
        print("Initializing JarvisAgent...")
        jarvis_agent = JarvisAgent()
        print("✓ JarvisAgent initialized successfully")
        
        print("=" * 60)
        print("Backend initialized successfully!")
        print("=" * 60)
        return True
        
    except ValueError as e:
        error_msg = f"Configuration Error: {str(e)}"
        print(f"❌ {error_msg}")
        init_error_message = error_msg
        import traceback
        traceback.print_exc()
        return False
    except ImportError as e:
        error_msg = f"Import Error: {str(e)}"
        print(f"❌ {error_msg}")
        init_error_message = error_msg
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        error_msg = f"Initialization Error: {str(e)}"
        print(f"❌ {error_msg}")
        init_error_message = error_msg
        import traceback
        traceback.print_exc()
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize backend on startup"""
    print("=" * 60)
    print("Starting Jarvis Backend API...")
    print("=" * 60)
    success = initialize_backend()
    if not success:
        print("=" * 60)
        print("⚠️  WARNING: Backend initialization failed!")
        print("Check logs above for details.")
        print("The API will still start, but chat endpoints will return errors.")
        print("=" * 60)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str

class BriefingResponse(BaseModel):
    date: str
    summary: dict
    reviews_due: list
    contact_gaps: list
    upcoming_milestones: list
    life_events: list
    unresolved_concerns: list
    overdue_actions: list
    overdue_follow_ups: list

# Health check
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Jarvis API",
        "agent_ready": jarvis_agent is not None
    }

@app.get("/health")
async def health():
    """Health check with detailed status"""
    status = {
        "status": "healthy" if jarvis_agent is not None else "unhealthy",
        "agent_initialized": jarvis_agent is not None,
        "compliance_tracker_initialized": compliance_tracker is not None,
    }
    
    # Check environment variables
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "set" if os.getenv("AZURE_OPENAI_ENDPOINT") else "missing",
        "AZURE_OPENAI_API_KEY": "set" if os.getenv("AZURE_OPENAI_API_KEY") else "missing",
        "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT", "not set"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "not set"),
    }
    status["environment_variables"] = env_vars
    
    # Check files
    files = {
        "mock_data.json": "exists" if Path("mock_data.json").exists() else "missing",
        "chroma_db": "exists" if Path("chroma_db").exists() else "missing",
    }
    status["files"] = files
    
    # Add error message if initialization failed
    if init_error_message:
        status["initialization_error"] = init_error_message
    
    return status

@app.post("/api/reinitialize")
async def reinitialize():
    """Manually retry backend initialization"""
    global jarvis_agent, compliance_tracker, init_error_message
    print("=" * 60)
    print("Manual reinitialization requested...")
    print("=" * 60)
    success = initialize_backend()
    if success:
        return {
            "status": "success",
            "message": "Backend reinitialized successfully",
            "agent_ready": jarvis_agent is not None
        }
    else:
        return {
            "status": "failed",
            "message": "Reinitialization failed",
            "error": init_error_message,
            "agent_ready": jarvis_agent is not None
        }

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with Jarvis"""
    if not jarvis_agent:
        error_detail = "Jarvis agent not initialized"
        if init_error_message:
            error_detail += f". Error: {init_error_message}"
        else:
            error_detail += ". Check /health endpoint for details. Try POST /api/reinitialize to retry."
        raise HTTPException(status_code=503, detail=error_detail)
    
    try:
        response = jarvis_agent.chat(request.message)
        return ChatResponse(response=response, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Daily briefing endpoint
@app.get("/api/briefing", response_model=BriefingResponse)
async def get_briefing():
    """Get daily briefing"""
    if not compliance_tracker:
        raise HTTPException(status_code=503, detail="Compliance tracker not initialized")
    
    try:
        briefing = compliance_tracker.get_daily_briefing()
        return BriefingResponse(
            date=briefing.get("date", ""),
            summary=briefing.get("summary", {}),
            reviews_due=briefing.get("reviews_due", []),
            contact_gaps=briefing.get("contact_gaps", []),
            upcoming_milestones=briefing.get("upcoming_milestones", []),
            life_events=briefing.get("life_events", []),
            unresolved_concerns=briefing.get("unresolved_concerns", []),
            overdue_actions=briefing.get("overdue_actions", []),
            overdue_follow_ups=briefing.get("overdue_follow_ups", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting briefing: {str(e)}")

# Search clients endpoint
@app.post("/api/search")
async def search_clients(query: str):
    """Search for clients"""
    if not jarvis_agent:
        raise HTTPException(status_code=503, detail="Jarvis agent not initialized")
    
    try:
        results = jarvis_agent.vector_store.search(query, k=10)
        return {
            "results": [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

