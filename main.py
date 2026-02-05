"""Main entry point for AdvisoryAI Jarvis"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import LangGraph version, fallback to simple version
try:
    from jarvis_graph import JarvisAgent
except (ImportError, Exception) as e:
    print(f"Warning: Could not import LangGraph version: {e}")
    print("Falling back to simplified version...")
    from jarvis_simple import JarvisAgentSimple as JarvisAgent
from data_generator import generate_all_clients
from vector_store import ClientVectorStore
from compliance_tracker import ComplianceTracker
import json


def initialize_system():
    """Initialize the system with mock data if needed"""
    data_file = Path("mock_data.json")
    vector_store_path = Path("chroma_db")
    
    # Generate mock data if it doesn't exist
    if not data_file.exists():
        print("Generating mock client data...")
        data = generate_all_clients(200)
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Generated {data['total_clients']} clients and {data['total_meetings']} meetings")
    
    # Initialize vector store
    print("Initializing vector store...")
    vector_store = ClientVectorStore()
    
    # Check if vector store is empty
    if not vector_store_path.exists() or len(os.listdir(vector_store_path)) == 0:
        print("Indexing client data...")
        vector_store.load_client_data(str(data_file))
        print("✓ Vector store initialized")
    else:
        print("✓ Vector store already exists")
    
    return vector_store


def main():
    """Main CLI interface"""
    print("=" * 60)
    print("AdvisoryAI Jarvis - Proactive Assistant for Financial Advisors")
    print("=" * 60)
    print()
    
    # Initialize system
    try:
        initialize_system()
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Make sure you have set OPENAI_API_KEY in your .env file")
        return
    
    # Initialize agent
    print("\nInitializing Jarvis...")
    try:
        agent = JarvisAgent()
        print("✓ Jarvis ready!")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure you have set OPENAI_API_KEY in your .env file")
        return
    
    print("\n" + "=" * 60)
    print("Jarvis: Hello! I'm Jarvis, your proactive assistant.")
    print("I help you stay on top of client relationships, compliance, and opportunities.")
    print("\nTry asking me:")
    print("  • 'What needs my attention today?'")
    print("  • 'Show me reviews due'")
    print("  • 'Who has upcoming milestone birthdays?'")
    print("  • 'Tell me about clients worried about inheritance tax'")
    print("  • 'What life events need attention?'")
    print("\nType 'quit' to exit, 'briefing' for daily briefing")
    print("=" * 60 + "\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "bye", "q"]:
                print("\nJarvis: Goodbye! Have a productive day!")
                break
            
            if user_input.lower() in ["briefing", "brief", "daily"]:
                user_input = "What needs my attention today? Give me a comprehensive daily briefing."
            
            print("\nJarvis: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nJarvis: Goodbye! Have a productive day!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()

