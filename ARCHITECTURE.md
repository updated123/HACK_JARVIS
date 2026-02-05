# Architecture Overview

## System Design

### Core Components

1. **Data Layer**
   - `data_generator.py`: Generates realistic mock data (200 clients, meeting notes, life events)
   - `mock_data.json`: Structured JSON storage for client information

2. **Vector Store Layer**
   - `vector_store.py`: ChromaDB integration for semantic search
   - Embeds client profiles, meeting transcripts, and notes
   - Enables natural language queries across all client data

3. **Compliance Layer**
   - `compliance_tracker.py`: Tracks FCA Consumer Duty requirements
   - Monitors annual reviews, contact gaps, milestones, life events
   - Generates proactive alerts and daily briefings

4. **AI Agent Layer**
   - `jarvis_graph.py`: LangGraph workflow for the chatbot
   - Tool integration for search and compliance queries
   - Natural language understanding and response generation

5. **Interface Layer**
   - `app.py`: Streamlit web interface
   - `main.py`: CLI interface
   - Both connect to the same Jarvis agent

## LangGraph Workflow

```
User Query
    ↓
Agent Node (LLM with tools)
    ↓
[Has tool calls?]
    ├─ Yes → Tool Node (Execute tools)
    │          ↓
    │       Agent Node (Process results)
    │          ↓
    └─ No → END (Return response)
```

### State Management

The `AgentState` TypedDict tracks:
- `messages`: Conversation history
- `current_query`: Current user question
- `context`: Additional context
- `compliance_data`: Cached compliance information

### Tools Available

1. **search_clients**: Semantic search across all client data
2. **get_client_details**: Get specific client information
3. **get_daily_briefing**: Comprehensive daily overview
4. **get_reviews_due**: Annual review tracking
5. **get_upcoming_milestones**: Milestone birthday alerts
6. **get_life_events**: Life event tracking
7. **get_unresolved_concerns**: Client concern follow-ups

## Data Flow

### Initialization
1. Generate/load mock data (`mock_data.json`)
2. Initialize vector store (ChromaDB)
3. Embed and index all client documents
4. Initialize compliance tracker
5. Load LangGraph agent with tools

### Query Processing
1. User asks question (natural language)
2. Agent determines if tools needed
3. Tools execute (search, compliance checks)
4. Agent synthesizes results
5. Response returned to user

### Vector Search
- Client profiles → embedded → stored in ChromaDB
- Meeting transcripts → chunked → embedded → stored
- Queries → embedded → similarity search → relevant chunks returned

## Key Features

### Proactive Briefings
- Daily overview of priorities
- Reviews due (with days overdue)
- Contact gaps (FCA compliance)
- Upcoming opportunities (milestones, life events)
- Overdue actions and follow-ups

### Semantic Search
- Natural language queries
- Finds relevant information across all clients
- Context-aware responses
- References specific clients and dates

### Compliance Tracking
- Automatic deadline monitoring
- Consumer Duty compliance
- Review scheduling
- Contact gap identification

## Technology Stack

- **LangChain**: LLM orchestration and tool integration
- **LangGraph**: Workflow management and state handling
- **ChromaDB**: Vector database for semantic search
- **OpenAI GPT-4**: Language model for understanding and generation
- **Streamlit**: Web interface
- **Python**: Core language

## Scalability Considerations

### Current Implementation (Hackathon)
- Mock data (200 clients)
- In-memory processing
- Single vector store instance
- No persistence of chat history

### Production Enhancements
- Database integration (PostgreSQL/MongoDB)
- Distributed vector stores
- Chat history persistence
- Multi-advisor support
- Real-time updates from CRM
- Caching layer for frequent queries
- Background job processing for compliance checks

## Security & Compliance

### Current
- API keys in `.env` file (not committed)
- Mock data only (no real client data)

### Production Requirements
- Encrypted data storage
- Access controls and authentication
- Audit logging
- GDPR compliance
- FCA regulatory compliance
- Data retention policies

## Performance Optimizations

1. **Vector Store Caching**: Frequently accessed clients cached
2. **Compliance Data Caching**: Daily briefings cached until refresh
3. **Chunking Strategy**: Optimal chunk sizes for retrieval
4. **Tool Selection**: Smart routing to appropriate tools

## Error Handling

- Graceful degradation if vector store unavailable
- Fallback responses if LLM fails
- Clear error messages for users
- Logging for debugging

## Testing Strategy

1. **Unit Tests**: Individual components (compliance tracker, vector store)
2. **Integration Tests**: End-to-end query processing
3. **Mock Data Validation**: Ensure realistic scenarios
4. **Tool Testing**: Verify each tool works correctly

## Future Enhancements

1. **Real Integrations**
   - CRM systems (Intelliflo, etc.)
   - Portfolio platforms
   - Calendar systems
   - Email/SMS notifications

2. **Advanced Features**
   - Document generation
   - Automated scheduling
   - Predictive analytics
   - Client sentiment analysis
   - Portfolio recommendations

3. **UI Improvements**
   - Mobile app
   - Voice interface
   - Dashboard visualizations
   - Export capabilities

