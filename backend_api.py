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

def initialize_backend():
    """Initialize backend services"""
    global jarvis_agent, compliance_tracker
    try:
        from jarvis_graph import JarvisAgent
        from compliance_tracker import ComplianceTracker
        from data_generator import generate_all_clients
        
        # Generate mock data if needed
        data_file = Path("mock_data.json")
        if not data_file.exists():
            print("Generating mock data...")
            generate_all_clients()
            print("Mock data generated successfully")
        
        # Set credentials from environment variables (Render sets these)
        import config
        config.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        config.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        config.AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        config.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        jarvis_agent = JarvisAgent()
        compliance_tracker = ComplianceTracker()
        print("Backend initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing backend: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_backend()

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
    return {
        "status": "healthy",
        "agent_initialized": jarvis_agent is not None
    }

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with Jarvis"""
    if not jarvis_agent:
        raise HTTPException(status_code=503, detail="Jarvis agent not initialized")
    
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

