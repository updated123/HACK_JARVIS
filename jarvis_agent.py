"""LangGraph workflow for Jarvis proactive chatbot"""
from typing import TypedDict, Annotated, Sequence
# Try new import path first, fallback to old path
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    try:
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        # Last resort: try langchain.chat_models
        from langchain_core.prompts.chat import ChatPromptTemplate
        MessagesPlaceholder = None  # Will handle this separately if needed

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jarvis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Azure OpenAI imports
try:
    from langchain_openai import AzureChatOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    print("Warning: langchain-openai not installed. Install with: pip install langchain-openai")

# LangGraph imports - try different versions
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    # Try to import ToolNode from different locations
    try:
        from langgraph.prebuilt import ToolNode
    except ImportError:
        try:
            from langgraph.prebuilt.tool_node import ToolNode
        except ImportError:
            # Create a simple tool node ourselves
            ToolNode = None
except ImportError as e:
    raise ImportError(f"Failed to import LangGraph: {e}. Please install with: pip install langgraph")

from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_API_KEY
)
from vector_store import ClientVectorStore
from compliance_tracker import ComplianceTracker


class AgentState(TypedDict):
    """State for the Jarvis agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_query: str
    context: dict
    compliance_data: dict


class JarvisAgent:
    """Proactive chatbot agent for financial advisors"""
    
    def __init__(self):
        logger.info("Initializing JarvisAgent...")
        # Use Azure OpenAI
        if not AZURE_OPENAI_AVAILABLE:
            logger.error("Azure OpenAI not available")
            raise RuntimeError("Azure OpenAI not available. Install with: pip install langchain-openai")
        
        try:
            logger.info(f"Initializing Azure OpenAI LLM with deployment: {AZURE_OPENAI_DEPLOYMENT}")
            self.llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
                api_key=AZURE_OPENAI_API_KEY,
                temperature=0
            )
            logger.info("Azure OpenAI LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {e}")
        
        logger.info("Initializing vector store...")
        self.vector_store = ClientVectorStore()
        logger.info("Vector store initialized")
        
        logger.info("Initializing compliance tracker...")
        self.compliance_tracker = ComplianceTracker()
        logger.info("Compliance tracker initialized")
        
        # Rate limit and token tracking (Azure has higher limits)
        self.rate_limit_threshold = 50000  # Azure has much higher limits
        self.current_token_count = 0
        self.rate_limit_window_start = datetime.now()
        self.fallback_mode = False
        self.rate_limit_errors = 0
        
        # Define tools - use StructuredTool for Azure OpenAI compatibility
        # Tools that need no parameters
        def make_no_param_tool(func, name, desc):
            return StructuredTool.from_function(
                func=func,
                name=name,
                description=desc
            )
        
        # Tools that need a query parameter
        def make_query_tool(func, name, desc):
            return StructuredTool.from_function(
                func=lambda query: func(query if query else ""),
                name=name,
                description=desc
            )
        
        # Optimized tool descriptions - shorter to reduce token usage
        self.tools = [
            make_query_tool(self._search_clients, "search_clients", "Search clients"),
            make_query_tool(self._get_client_details, "get_client_details", "Get client details"),
            make_no_param_tool(lambda: self._get_daily_briefing(""), "get_daily_briefing", "Daily briefing"),
            make_no_param_tool(lambda: self._get_reviews_due(""), "get_reviews_due", "Reviews due"),
            make_no_param_tool(lambda: self._get_upcoming_milestones(""), "get_upcoming_milestones", "Milestones"),
            make_no_param_tool(lambda: self._get_life_events(""), "get_life_events", "Life events"),
            make_no_param_tool(lambda: self._get_unresolved_concerns(""), "get_unresolved_concerns", "Concerns"),
            make_no_param_tool(lambda: self._get_clients_underweight_equities(""), "get_clients_underweight_equities", "Underweight equities"),
            make_no_param_tool(lambda: self._get_clients_with_isa_allowance(""), "get_clients_with_isa_allowance", "ISA allowance"),
            make_no_param_tool(lambda: self._get_clients_with_annual_allowance(""), "get_clients_with_annual_allowance", "Annual allowance"),
            make_no_param_tool(lambda: self._get_clients_with_cash_excess(""), "get_clients_with_cash_excess", "Cash excess"),
            make_no_param_tool(lambda: self._get_clients_retirement_trajectory_issues(""), "get_clients_retirement_trajectory_issues", "Retirement issues"),
            make_no_param_tool(lambda: self._get_clients_with_protection_gaps(""), "get_clients_with_protection_gaps", "Protection gaps"),
            make_no_param_tool(lambda: self._get_retired_clients_high_withdrawal(""), "get_retired_clients_high_withdrawal", "High withdrawal"),
            make_no_param_tool(lambda: self._get_clients_reviews_overdue_12_months(""), "get_clients_reviews_overdue_12_months", "Overdue 12+mo"),
            make_no_param_tool(lambda: self._get_business_owners(""), "get_business_owners", "Business owners"),
            make_no_param_tool(lambda: self._get_clients_university_planning(""), "get_clients_university_planning", "University planning"),
            make_query_tool(self._find_similar_clients, "find_similar_clients", "Similar clients"),
            make_no_param_tool(lambda: self._get_hnw_clients_no_estate_planning(""), "get_hnw_clients_no_estate_planning", "HNW no estate"),
            make_no_param_tool(lambda: self._get_pension_clients_for_cashflow(""), "get_pension_clients_for_cashflow", "Pension cashflow"),
            make_no_param_tool(lambda: self._get_clients_investments_no_protection(""), "get_clients_investments_no_protection", "No protection"),
            make_no_param_tool(lambda: self._get_business_owners_no_exit_planning(""), "get_business_owners_no_exit_planning", "No exit plan"),
            make_no_param_tool(lambda: self._get_clients_birthdays_this_month(""), "get_clients_birthdays_this_month", "This month bdays"),
            make_query_tool(self._get_recommendations_for_client, "get_recommendations_for_client", "Recommendations"),
            make_query_tool(self._search_conversation_wording, "search_conversation_wording", "Search wording"),
            make_query_tool(self._get_clients_recommended_platform, "get_clients_recommended_platform", "Platform clients"),
            make_query_tool(self._get_conversations_about_topic, "get_conversations_about_topic", "Topic conversations"),
            make_query_tool(self._get_discussion_summary, "get_discussion_summary", "Discussion summary"),
            make_no_param_tool(lambda: self._get_documents_waiting(""), "get_documents_waiting", "Waiting docs"),
            make_no_param_tool(lambda: self._get_promises_made(""), "get_promises_made", "Promises"),
            make_no_param_tool(lambda: self._get_concerns_this_month(""), "get_concerns_this_month", "This month concerns"),
            make_no_param_tool(lambda: self._get_service_usage_analysis(""), "get_service_usage_analysis", "Service usage"),
            make_no_param_tool(lambda: self._get_conversion_rates(""), "get_conversion_rates", "Conversion rates"),
            make_no_param_tool(lambda: self._get_book_demographics(""), "get_book_demographics", "Demographics"),
            make_no_param_tool(lambda: self._get_revenue_time_analysis(""), "get_revenue_time_analysis", "Revenue/time"),
            make_no_param_tool(lambda: self._get_satisfied_client_patterns(""), "get_satisfied_client_patterns", "Satisfied patterns"),
            make_no_param_tool(lambda: self._get_recommendation_pushback(""), "get_recommendation_pushback", "Pushback"),
            make_query_tool(self._get_similar_circumstances_cases, "get_similar_circumstances_cases", "Similar cases"),
            make_no_param_tool(lambda: self._get_life_events_triggering_implementation(""), "get_life_events_triggering_implementation", "Life events impl"),
            make_query_tool(self._draft_follow_up_email, "draft_follow_up_email", "Draft email"),
            make_no_param_tool(lambda: self._get_waiting_on_clients(""), "get_waiting_on_clients", "Waiting on"),
            make_no_param_tool(lambda: self._get_open_action_items(""), "get_open_action_items", "Action items"),
            make_no_param_tool(lambda: self._get_overdue_follow_ups(""), "get_overdue_follow_ups", "Overdue followups"),
            # Scenario modeling and analysis tools
            make_query_tool(self._analyze_interest_rate_impact, "analyze_interest_rate_impact", "Interest rate impact"),
            make_query_tool(self._analyze_market_correction_exposure, "analyze_market_correction_exposure", "Market correction exposure"),
            make_query_tool(self._model_retirement_scenario, "model_retirement_scenario", "Retirement scenario"),
            make_query_tool(self._model_long_term_care_scenario, "model_long_term_care_scenario", "Long-term care scenario"),
            make_no_param_tool(lambda: self._get_business_owners_rd_tax_credit(""), "get_business_owners_rd_tax_credit", "R&D tax credit"),
            # Proactive Intelligence Features
            make_no_param_tool(lambda: self._get_priority_scored_opportunities(""), "get_priority_scored_opportunities", "Get top opportunities ranked by priority score and impact"),
            make_no_param_tool(lambda: self._get_relationship_health_dashboard(""), "get_relationship_health", "Get relationship health dashboard with risk analysis"),
            make_no_param_tool(lambda: self._get_opportunity_window_alerts(""), "get_opportunity_alerts", "Get time-sensitive opportunity alerts with deadlines"),
            make_no_param_tool(lambda: self._get_predictive_risk_detection(""), "get_predictive_risks", "Get predictive risk analysis across all clients"),
            make_no_param_tool(lambda: self._get_contextual_morning_briefing(""), "get_contextual_briefing", "Get personalized contextual morning briefing"),
            make_no_param_tool(lambda: self._get_client_clusters(""), "get_client_clusters", "Get client clusters and pattern recognition"),
            make_no_param_tool(lambda: self._get_value_demonstration_summary(""), "get_value_summary", "Get value demonstration summary showing delivered value")
        ]
        
        # Create tool node if available, otherwise use custom implementation
        logger.info(f"Creating tool node with {len(self.tools)} tools...")
        if ToolNode is not None:
            self.tool_node = ToolNode(self.tools)
            logger.info("Using LangGraph ToolNode")
        else:
            self.tool_node = self._create_tool_node()
            logger.info("Using custom tool node")
        
        # Build graph
        logger.info("Building LangGraph workflow...")
        self.graph = self._build_graph()
        logger.info("JarvisAgent initialization complete")
    
    def _create_tool_node(self):
        """Create a custom tool node if ToolNode is not available"""
        def tool_node(state: AgentState) -> AgentState:
            """Execute tools based on AI message tool calls"""
            logger.info("tool_node called")
            last_message = state["messages"][-1]
            logger.debug(f"Last message type: {type(last_message)}")
            
            if isinstance(last_message, AIMessage):
                tool_calls = getattr(last_message, 'tool_calls', None) or []
                logger.info(f"Found {len(tool_calls)} tool calls")
                
                if tool_calls:
                    from langchain_core.messages import ToolMessage
                    tool_messages = []
                    
                    for idx, tool_call in enumerate(tool_calls):
                        logger.info(f"Processing tool call {idx+1}/{len(tool_calls)}")
                        # Handle different tool_call formats
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get("name") or tool_call.get("id", "")
                            tool_input = tool_call.get("args", {})
                            tool_call_id = tool_call.get("id") or tool_name
                        else:
                            # Handle object format
                            tool_name = getattr(tool_call, "name", getattr(tool_call, "id", ""))
                            tool_input = getattr(tool_call, "args", {})
                            tool_call_id = getattr(tool_call, "id", tool_name)
                        
                        # Find the tool
                        tool_func = None
                        for tool in self.tools:
                            if tool.name == tool_name:
                                tool_func = tool.func
                                break
                        
                        if tool_func:
                            logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                            try:
                                # Handle Azure OpenAI's tool calling format
                                # Azure OpenAI may pass empty dict {} or dict with parameters
                                import inspect
                                sig = inspect.signature(tool_func)
                                params = list(sig.parameters.keys())
                                
                                if isinstance(tool_input, dict):
                                    if not tool_input or tool_input == {}:
                                        # Empty dict - tool has no parameters
                                        if len(params) == 0:
                                            result = tool_func()
                                        elif len(params) == 1 and params[0] == "query":
                                            # Tool expects query but got empty dict
                                            result = tool_func("")
                                        else:
                                            result = tool_func()
                                    else:
                                        # Non-empty dict
                                        if len(params) == 1 and params[0] == "query":
                                            # Extract query from dict (handle various key names)
                                            query_val = tool_input.get("query") or tool_input.get("client_name") or tool_input.get("text") or ""
                                            result = tool_func(query_val)
                                        else:
                                            # Try to match parameters
                                            filtered_args = {k: v for k, v in tool_input.items() if k in params}
                                            if filtered_args:
                                                result = tool_func(**filtered_args)
                                            else:
                                                # Fallback: try calling with first param if single param
                                                if len(params) == 1:
                                                    result = tool_func(list(tool_input.values())[0] if tool_input else "")
                                                else:
                                                    result = tool_func()
                                elif isinstance(tool_input, str):
                                    # String input - check if tool expects query
                                    if len(params) == 1 and params[0] == "query":
                                        result = tool_func(tool_input)
                                    elif len(params) == 0:
                                        result = tool_func()
                                    else:
                                        result = tool_func(**{params[0]: tool_input})
                                elif not tool_input or tool_input == {}:
                                    # No input - call without args
                                    result = tool_func()
                                else:
                                    # Unknown format - try to convert
                                    if len(params) == 1:
                                        result = tool_func(str(tool_input))
                                    else:
                                        result = tool_func()
                                
                                logger.info(f"Tool {tool_name} executed successfully, result length: {len(str(result))}")
                                tool_messages.append(ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_call_id
                                ))
                            except Exception as e:
                                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                                tool_messages.append(ToolMessage(
                                    content=f"Error executing {tool_name}: {str(e)}",
                                    tool_call_id=tool_call_id
                                ))
                    
                    logger.info(f"Returning {len(tool_messages)} tool messages")
                    return {"messages": tool_messages}
            
            logger.warning("No tool calls found in last message")
            return {"messages": []}
        
        return tool_node
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self._should_use_tools,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        # Compile graph without checkpointer (we don't need state persistence for this use case)
        # If you need state persistence later, you can add a checkpointer with proper config
        try:
            return workflow.compile()
        except Exception as e:
            # Try with explicit None checkpointer
            try:
                return workflow.compile(checkpointer=None)
            except Exception as e2:
                raise RuntimeError(f"Failed to compile graph: {e2}")
    
    def _should_use_tools(self, state: AgentState) -> str:
        """Determine if tools should be used"""
        logger.debug("_should_use_tools called")
        last_message = state["messages"][-1]
        
        # If the last message is from AI and has tool calls, use tools
        if isinstance(last_message, AIMessage):
            # Check if message has tool calls
            tool_calls = getattr(last_message, 'tool_calls', None)
            logger.debug(f"Last message is AIMessage, tool_calls: {tool_calls}")
            if tool_calls and len(tool_calls) > 0:
                logger.info(f"Routing to tools node with {len(tool_calls)} tool calls")
                return "tools"
        
        logger.info("Routing to end (no tool calls)")
        return "end"
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 characters per token"""
        return len(text) // 4
    
    def _check_rate_limit(self, estimated_tokens: int) -> bool:
        """Check if we're approaching rate limits"""
        # Reset counter if a minute has passed
        time_diff = (datetime.now() - self.rate_limit_window_start).total_seconds()
        if time_diff > 60:
            self.current_token_count = 0
            self.rate_limit_window_start = datetime.now()
            self.fallback_mode = False
        
        # Check if adding these tokens would exceed threshold
        if self.current_token_count + estimated_tokens > self.rate_limit_threshold:
            logger.warning(f"Token limit approaching: {self.current_token_count + estimated_tokens} > {self.rate_limit_threshold}")
            self.fallback_mode = True
            return False
        
        self.current_token_count += estimated_tokens
        return True
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent node that processes messages"""
        logger.info("_agent_node called")
        messages = state["messages"]
        logger.debug(f"Processing {len(messages)} messages")
        
        # Enhanced system prompt - comprehensive coverage with emphasis on agentic tool selection
        system_prompt = SystemMessage(content="""You are Jarvis, a proactive AI assistant for UK Financial Advisors using an AGENTIC WORKFLOW.

AGENTIC WORKFLOW - YOU DECIDE WHICH TOOLS TO USE:
- Analyze the user's question carefully
- Select the appropriate tool(s) from the available tools
- You can call multiple tools if needed to gather comprehensive information
- Process tool results and provide a thoughtful, synthesized answer
- Use your intelligence to determine the best approach - don't rely on simple keyword matching

AVAILABLE TOOLS (You decide when to use each):
1. Investment Analysis: get_clients_underweight_equities, get_clients_with_isa_allowance, get_clients_with_annual_allowance, get_clients_with_cash_excess, get_clients_retirement_trajectory_issues, get_clients_with_protection_gaps, get_retired_clients_high_withdrawal, analyze_interest_rate_impact, analyze_market_correction_exposure
2. Proactive Identification: get_clients_reviews_overdue_12_months, get_business_owners, get_business_owners_rd_tax_credit, get_clients_university_planning, find_similar_clients, get_hnw_clients_no_estate_planning, get_clients_birthdays_this_month
3. Compliance & Documentation: get_recommendations_for_client, search_conversation_wording, get_clients_recommended_platform, get_conversations_about_topic, get_discussion_summary, get_documents_waiting, get_promises_made
4. Business Analytics: get_service_usage_analysis, get_conversion_rates, get_book_demographics, get_revenue_time_analysis, get_satisfied_client_patterns, get_recommendation_pushback, get_similar_circumstances_cases
5. Follow-ups & Actions: draft_follow_up_email, get_waiting_on_clients, get_open_action_items, get_overdue_follow_ups, get_life_events_triggering_implementation
6. General: search_clients, get_client_details, get_daily_briefing, get_reviews_due, get_upcoming_milestones, get_life_events, get_unresolved_concerns
7. Scenario Modeling: model_retirement_scenario, model_long_term_care_scenario

CONVERSATION CONTEXT:
- You maintain full conversation history - reference previous questions and answers
- Build on previous context - if asked about a client, remember it for follow-up questions
- Use "we discussed earlier" when referencing previous conversation

GUIDELINES:
- ALWAYS use tools to gather data - never guess or make up information
- You can call multiple tools in sequence if needed
- Synthesize tool results into a comprehensive, actionable answer
- Be concise but thorough - prioritize urgent/important items first
- Provide specific client names, dates, and numbers from tool results
- Connect related information across different tools when relevant

INNOVATION & IMPACT:
- Proactively suggest related opportunities based on tool results
- Identify patterns and trends across the client base
- Highlight time-sensitive opportunities
- Provide strategic insights beyond just answering the question

PROACTIVE INTELLIGENCE FEATURES (NEW):
1. Priority Scoring: get_priority_scored_opportunities - Automatically ranks opportunities by impact, urgency, and value. Use when advisor asks "what should I focus on" or "top priorities"
2. Relationship Health: get_relationship_health - Dashboard showing client relationship health, risk levels, and value at risk. Use for "relationship health" or "at-risk clients"
3. Opportunity Alerts: get_opportunity_alerts - Time-sensitive opportunities with deadlines. Use for "deadlines" or "expiring opportunities"
4. Predictive Risks: get_predictive_risks - Identifies problems before they become critical. Use for "risks" or "potential problems"
5. Contextual Briefing: get_contextual_briefing - Personalized morning briefing with top priorities. Use for "morning briefing" or "what should I do today"
6. Client Clusters: get_client_clusters - Groups similar clients and identifies patterns. Use for "similar clients" or "patterns"
7. Value Summary: get_value_summary - Quantifies value delivered to clients. Use for "value delivered" or "demonstrate value"

When to use proactive features:
- Advisor asks "what should I focus on" → Use get_priority_scored_opportunities
- Advisor asks "who needs attention" → Use get_relationship_health
- Advisor asks "what deadlines" → Use get_opportunity_alerts
- Advisor asks "what risks" → Use get_predictive_risks
- Advisor asks "morning briefing" → Use get_contextual_briefing
- Advisor asks "show me patterns" → Use get_client_clusters
- Advisor asks "value delivered" → Use get_value_summary

ALWAYS prioritize proactive insights over reactive answers!""")
        
        # Prepare messages with system prompt
        prompt_messages = [system_prompt] + list(messages)
        logger.debug(f"Prepared {len(prompt_messages)} messages for LLM")
        
        # Estimate tokens for this request
        prompt_text = " ".join([str(msg.content) if hasattr(msg, 'content') else str(msg) for msg in prompt_messages])
        estimated_tokens = self._estimate_tokens(prompt_text) + 2000  # Add buffer for tool descriptions
        
        # Check rate limits before making request
        if not self._check_rate_limit(estimated_tokens):
            logger.warning("Rate limit threshold reached - using fallback mode")
            self.fallback_mode = True
            raise RuntimeError("RATE_LIMIT_EXCEEDED")
        
        # Bind tools to LLM
        logger.debug(f"Binding {len(self.tools)} tools to LLM")
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Get response with error handling and retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Invoking LLM... (attempt {attempt + 1}/{max_retries})")
                response = llm_with_tools.invoke(prompt_messages)
                logger.info(f"LLM response received. Type: {type(response)}")
                
                # Success - reset fallback mode
                self.fallback_mode = False
                self.rate_limit_errors = 0
                
                # Check for tool calls
                tool_calls = getattr(response, 'tool_calls', None)
                if tool_calls:
                    logger.info(f"LLM made {len(tool_calls)} tool calls: {[tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in tool_calls]}")
                else:
                    logger.info("LLM response has no tool calls - final answer")
                    logger.debug(f"Response content: {response.content[:200] if hasattr(response, 'content') else 'N/A'}")
                
                return {"messages": [response]}
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"LLM invocation error (attempt {attempt + 1}): {e}")
                
                # Check for rate limit errors (429, quota, etc.)
                if any(keyword in error_msg for keyword in ["quota", "429", "insufficient_quota", "rate limit", "too many requests", "413", "too large"]):
                    self.rate_limit_errors += 1
                    self.fallback_mode = True
                    
                    if attempt < max_retries - 1:
                        # Wait a bit before retry (exponential backoff)
                        import time
                        wait_time = (attempt + 1) * 2
                        logger.info(f"Rate limit hit, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning("Rate limit exceeded after retries - will trigger fallback")
                        raise RuntimeError("RATE_LIMIT_EXCEEDED")
                else:
                    logger.error(f"Unexpected error in agent node: {e}", exc_info=True)
                    raise
        
        # Should not reach here, but just in case
        raise RuntimeError("RATE_LIMIT_EXCEEDED")
    
    # Tool functions
    def _search_clients(self, query: str) -> str:
        """Search for client information"""
        results = self.vector_store.search(query, k=5)
        
        if not results:
            return "No relevant client information found."
        
        response = f"Found {len(results)} relevant results:\n\n"
        for i, doc in enumerate(results, 1):
            client_name = doc.metadata.get("client_name", "Unknown")
            doc_type = doc.metadata.get("type", "unknown")
            response += f"{i}. {client_name} ({doc_type})\n"
            response += f"   {doc.page_content[:300]}...\n\n"
        
        return response
    
    def _get_client_details(self, client_name: str) -> str:
        """Get detailed information about a specific client"""
        results = self.vector_store.search_by_client(client_name, k=10)
        
        if not results:
            return f"No information found for client: {client_name}"
        
        # Group by type
        profile = None
        meetings = []
        
        for doc in results:
            if doc.metadata.get("type") == "client_profile":
                profile = doc
            elif doc.metadata.get("type") == "meeting_note":
                meetings.append(doc)
        
        response = f"Client Information: {client_name}\n"
        response += "=" * 50 + "\n\n"
        
        if profile:
            response += "Profile:\n"
            response += profile.page_content[:500] + "...\n\n"
        
        if meetings:
            response += f"Recent Meetings ({len(meetings)} found):\n"
            for meeting in meetings[:3]:  # Show top 3
                response += f"- {meeting.metadata.get('date')}: {meeting.page_content[:200]}...\n\n"
        
        return response
    
    def _get_daily_briefing(self, query: str = "") -> str:
        """Get daily briefing"""
        briefing = self.compliance_tracker.get_daily_briefing()
        summary = briefing["summary"]
        
        response = f"Daily Briefing - {briefing['date']}\n"
        response += "=" * 60 + "\n\n"
        
        response += "PRIORITY ITEMS:\n"
        response += f"  • Reviews Due: {summary['total_reviews_due']}\n"
        response += f"  • Contact Gaps: {summary['total_contact_gaps']}\n"
        response += f"  • Overdue Actions: {summary['total_overdue_actions']}\n"
        response += f"  • Overdue Follow-ups: {summary['total_overdue_follow_ups']}\n\n"
        
        response += "OPPORTUNITIES:\n"
        response += f"  • Upcoming Milestones: {summary['total_milestones']}\n"
        response += f"  • Life Events: {summary['total_life_events']}\n"
        response += f"  • Unresolved Concerns: {summary['total_concerns']}\n\n"
        
        # Top 5 reviews due
        if briefing["reviews_due"]:
            response += "TOP REVIEWS DUE:\n"
            for review in briefing["reviews_due"][:5]:
                response += f"  • {review['client_name']} - {review['days_overdue']} days overdue\n"
            response += "\n"
        
        # Top milestones
        if briefing["upcoming_milestones"]:
            response += "UPCOMING MILESTONES:\n"
            for milestone in briefing["upcoming_milestones"][:5]:
                response += f"  • {milestone['client_name']} turning {milestone['turning_age']} in {milestone['days_until']} days\n"
            response += "\n"
        
        return response
    
    def _get_reviews_due(self, query: str = "") -> str:
        """Get reviews due"""
        reviews = self.compliance_tracker.get_reviews_due()
        
        if not reviews:
            return "No reviews currently due. Great job staying on top of compliance!"
        
        response = f"Reviews Due: {len(reviews)}\n\n"
        for review in reviews[:10]:
            response += f"• {review['client_name']} - {review['days_overdue']} days overdue (Last review: {review['last_review']})\n"
        
        return response
    
    def _get_upcoming_milestones(self, query: str = "") -> str:
        """Get upcoming milestones"""
        milestones = self.compliance_tracker.get_upcoming_milestones()
        
        if not milestones:
            return "No upcoming milestone birthdays in the next 30 days."
        
        response = f"Upcoming Milestones: {len(milestones)}\n\n"
        for milestone in milestones:
            response += f"• {milestone['client_name']} turning {milestone['turning_age']} on {milestone['birthday_date']} ({milestone['days_until']} days)\n"
            response += f"  Opportunity: {milestone['opportunity']}\n\n"
        
        return response
    
    def _get_life_events(self, query: str = "") -> str:
        """Get life events"""
        events = self.compliance_tracker.get_life_events_requiring_attention()
        
        if not events:
            return "No recent or upcoming life events requiring attention."
        
        response = f"Life Events Requiring Attention: {len(events)}\n\n"
        for event in events:
            days_text = f"{abs(event['days_until'])} days ago" if event['days_until'] < 0 else f"in {event['days_until']} days"
            response += f"• {event['client_name']}: {event['event_type']} ({days_text})\n"
            response += f"  Action: {event['action_required']}\n\n"
        
        return response
    
    def _get_unresolved_concerns(self, query: str = "") -> str:
        """Get unresolved concerns"""
        concerns = self.compliance_tracker.get_unresolved_concerns()
        
        if not concerns:
            return "No unresolved client concerns."
        
        response = f"Unresolved Concerns: {len(concerns)}\n\n"
        for concern in concerns:
            response += f"• {concern['client_name']}: {concern['concern']} (Status: {concern['status']})\n"
            response += f"  First mentioned: {concern['first_mentioned']}\n"
            response += f"  Action: {concern['action_required']}\n\n"
        
        return response
    
    # Investment analysis tools
    def _get_clients_underweight_equities(self, query: str = "") -> str:
        """Find clients underweight in equities"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        underweight = []
        for client_id, client in data["clients"].items():
            current_equity = client.get("equity_allocation_percent", 0)
            target_min = client.get("target_equity_allocation_min", 0)
            
            if current_equity < target_min:
                underweight.append({
                    "name": client["name"],
                    "current_equity": current_equity,
                    "target_min": target_min,
                    "risk_profile": client["risk_profile"],
                    "time_horizon": client.get("time_horizon_years", 0),
                    "portfolio_value": client["portfolio_value_gbp"]
                })
        
        if not underweight:
            return "No clients found who are underweight in equities relative to their risk profile."
        
        response = f"Found {len(underweight)} clients underweight in equities:\n\n"
        for c in sorted(underweight, key=lambda x: x["target_min"] - x["current_equity"], reverse=True)[:20]:
            gap = c["target_min"] - c["current_equity"]
            response += f"• {c['name']}: {c['current_equity']:.1f}% equities (target: {c['target_min']}-{c.get('target_max', c['target_min'])}%)\n"
            response += f"  Risk: {c['risk_profile']}, Time horizon: {c['time_horizon_years']} years, Portfolio: £{c['portfolio_value']:,.0f}\n\n"
        
        return response
    
    def _get_clients_with_isa_allowance(self, query: str = "") -> str:
        """Get clients with ISA allowance available"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        clients_with_allowance = []
        for client_id, client in data["clients"].items():
            available = client.get("isa_allowance_available", 0)
            if available > 0:
                clients_with_allowance.append({
                    "name": client["name"],
                    "available": available,
                    "used": client.get("isa_allowance_used", 0)
                })
        
        if not clients_with_allowance:
            return "No clients found with ISA allowance still available."
        
        response = f"Found {len(clients_with_allowance)} clients with ISA allowance available:\n\n"
        for c in sorted(clients_with_allowance, key=lambda x: x["available"], reverse=True)[:20]:
            response += f"• {c['name']}: £{c['available']:,.0f} available (used: £{c['used']:,.0f})\n"
        
        return response
    
    def _get_clients_with_annual_allowance(self, query: str = "") -> str:
        """Get clients with annual allowance available"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        clients_with_allowance = []
        for client_id, client in data["clients"].items():
            available = client.get("annual_allowance_available", 0)
            if available > 0:
                clients_with_allowance.append({
                    "name": client["name"],
                    "available": available,
                    "limit": client.get("annual_allowance_limit", 60000),
                    "used": client.get("annual_allowance_used", 0)
                })
        
        if not clients_with_allowance:
            return "No clients found with annual allowance still available."
        
        response = f"Found {len(clients_with_allowance)} clients with annual allowance available:\n\n"
        for c in sorted(clients_with_allowance, key=lambda x: x["available"], reverse=True)[:20]:
            response += f"• {c['name']}: £{c['available']:,.0f} available (limit: £{c['limit']:,.0f}, used: £{c['used']:,.0f})\n"
        
        return response
    
    def _get_clients_with_cash_excess(self, query: str = "") -> str:
        """Find clients with cash excess above 6 months"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        excess_cash = []
        for client_id, client in data["clients"].items():
            cash_months = client.get("cash_excess_months", 0)
            if cash_months > 6:
                excess_cash.append({
                    "name": client["name"],
                    "cash_holdings": client.get("cash_holdings_gbp", 0),
                    "monthly_expenditure": client.get("monthly_expenditure_gbp", 0),
                    "cash_months": cash_months,
                    "portfolio_value": client["portfolio_value_gbp"]
                })
        
        if not excess_cash:
            return "No clients found with cash excess above 6 months expenditure."
        
        response = f"Found {len(excess_cash)} clients with cash excess above 6 months:\n\n"
        for c in sorted(excess_cash, key=lambda x: x["cash_months"], reverse=True)[:20]:
            response += f"• {c['name']}: £{c['cash_holdings']:,.0f} cash ({c['cash_months']:.1f} months expenditure)\n"
            response += f"  Monthly spend: £{c['monthly_expenditure']:,.0f}, Portfolio: £{c['portfolio_value']:,.0f}\n\n"
        
        return response
    
    def _get_clients_retirement_trajectory_issues(self, query: str = "") -> str:
        """Find clients with retirement trajectory issues"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        issues = []
        for client_id, client in data["clients"].items():
            if not client.get("is_retired", False):
                goal = client.get("retirement_income_goal_monthly", 0)
                if goal > 0:
                    # Simple check: if portfolio can't sustain goal at 4% withdrawal
                    portfolio = client["portfolio_value_gbp"]
                    sustainable_income = (portfolio * 0.04) / 12 if portfolio > 0 else 0
                    if sustainable_income < goal * 0.9:  # 10% buffer
                        issues.append({
                            "name": client["name"],
                            "age": client["age"],
                            "retirement_age": client.get("retirement_age", 65),
                            "goal": goal,
                            "sustainable": sustainable_income,
                            "portfolio": portfolio,
                            "gap": goal - sustainable_income
                        })
        
        if not issues:
            return "No clients found with retirement trajectory issues."
        
        response = f"Found {len(issues)} clients whose trajectory won't meet retirement goals:\n\n"
        for c in sorted(issues, key=lambda x: x["gap"], reverse=True)[:20]:
            response += f"• {c['name']} (age {c['age']}, retiring at {c['retirement_age']}):\n"
            response += f"  Goal: £{c['goal']:,.0f}/month, Sustainable: £{c['sustainable']:,.0f}/month\n"
            response += f"  Gap: £{c['gap']:,.0f}/month, Portfolio: £{c['portfolio']:,.0f}\n\n"
        
        return response
    
    def _get_clients_with_protection_gaps(self, query: str = "") -> str:
        """Find clients with protection gaps"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        gaps = []
        for client_id, client in data["clients"].items():
            if client.get("has_protection_gap", False):
                gaps.append({
                    "name": client["name"],
                    "dependents": client.get("num_dependents", 0),
                    "has_life": client.get("has_life_insurance", False),
                    "has_ci": client.get("has_critical_illness", False),
                    "has_ip": client.get("has_income_protection", False),
                    "marital_status": client.get("marital_status", "")
                })
        
        if not gaps:
            return "No clients found with protection gaps."
        
        response = f"Found {len(gaps)} clients with protection gaps:\n\n"
        for c in gaps[:20]:
            response += f"• {c['name']}: {c['dependents']} dependents, {c['marital_status']}\n"
            response += f"  Life: {'Yes' if c['has_life'] else 'No'}, CI: {'Yes' if c['has_ci'] else 'No'}, IP: {'Yes' if c['has_ip'] else 'No'}\n\n"
        
        return response
    
    def _get_retired_clients_high_withdrawal(self, query: str = "") -> str:
        """Get retired clients with high withdrawal rates"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        high_withdrawal = []
        for client_id, client in data["clients"].items():
            if client.get("is_retired", False):
                withdrawal_rate = client.get("withdrawal_rate_percent", 0)
                if withdrawal_rate > 4:
                    high_withdrawal.append({
                        "name": client["name"],
                        "age": client["age"],
                        "withdrawal_rate": withdrawal_rate,
                        "income": client.get("current_retirement_income_monthly", 0),
                        "portfolio": client["portfolio_value_gbp"]
                    })
        
        if not high_withdrawal:
            return "No retired clients found taking more than 4% withdrawal rates."
        
        response = f"Found {len(high_withdrawal)} retired clients with withdrawal rates >4%:\n\n"
        for c in sorted(high_withdrawal, key=lambda x: x["withdrawal_rate"], reverse=True)[:20]:
            response += f"• {c['name']} (age {c['age']}): {c['withdrawal_rate']:.2f}% withdrawal rate\n"
            response += f"  Income: £{c['income']:,.0f}/month, Portfolio: £{c['portfolio']:,.0f}\n\n"
        
        return response
    
    # Proactive query tools
    def _get_clients_reviews_overdue_12_months(self, query: str = "") -> str:
        """Get clients without review in 12+ months"""
        reviews = self.compliance_tracker.get_reviews_due()
        overdue_12m = [r for r in reviews if r["days_overdue"] >= 365]
        
        if not overdue_12m:
            return "No clients found who haven't had a review in over 12 months."
        
        response = f"Found {len(overdue_12m)} clients without review in 12+ months:\n\n"
        for r in sorted(overdue_12m, key=lambda x: x["days_overdue"], reverse=True)[:20]:
            response += f"• {r['client_name']}: {r['days_overdue']} days overdue (Last review: {r['last_review']})\n"
        
        return response
    
    def _get_business_owners(self, query: str = "") -> str:
        """Get business owner clients"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        business_owners = []
        for client_id, client in data["clients"].items():
            if client.get("is_business_owner", False):
                business_owners.append({
                    "name": client["name"],
                    "business_type": client.get("business_type", "Unknown"),
                    "age": client["age"],
                    "portfolio": client["portfolio_value_gbp"]
                })
        
        if not business_owners:
            return "No business owner clients found."
        
        response = f"Found {len(business_owners)} business owner clients:\n\n"
        for c in business_owners[:20]:
            response += f"• {c['name']}: {c['business_type']}, Age {c['age']}, Portfolio: £{c['portfolio']:,.0f}\n"
        
        return response
    
    def _get_clients_university_planning(self, query: str = "") -> str:
        """Find clients with children approaching university age"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        university_needs = []
        for client_id, client in data["clients"].items():
            children_ages = client.get("children_ages", [])
            approaching = [age for age in children_ages if 16 <= age <= 19]
            has_planning = client.get("has_education_planning", False)
            
            if approaching and not has_planning:
                university_needs.append({
                    "name": client["name"],
                    "children_ages": children_ages,
                    "approaching": approaching
                })
        
        if not university_needs:
            return "No clients found with children approaching university age without education planning."
        
        response = f"Found {len(university_needs)} clients needing university planning:\n\n"
        for c in university_needs[:20]:
            response += f"• {c['name']}: Children ages {c['children_ages']}, {len(c['approaching'])} approaching university age\n"
        
        return response
    
    def _find_similar_clients(self, query: str) -> str:
        """Find similar clients - uses search"""
        # Extract client name from query if provided
        results = self.vector_store.search(query, k=10)
        
        if not results:
            return "No similar clients found. Try providing more details about the client profile you're looking for."
        
        response = f"Found {len(results)} similar clients:\n\n"
        for i, doc in enumerate(results[:10], 1):
            client_name = doc.metadata.get("client_name", "Unknown")
            response += f"{i}. {client_name}\n"
            response += f"   {doc.page_content[:200]}...\n\n"
        
        return response
    
    def _get_hnw_clients_no_estate_planning(self, query: str = "") -> str:
        """Get HNW clients without estate planning"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        hnw_no_planning = []
        for client_id, client in data["clients"].items():
            if client.get("is_high_net_worth", False) and not client.get("has_estate_planning", False):
                hnw_no_planning.append({
                    "name": client["name"],
                    "portfolio": client["portfolio_value_gbp"],
                    "has_will": client.get("has_will", False),
                    "has_trust": client.get("has_trust", False)
                })
        
        if not hnw_no_planning:
            return "No high-net-worth clients found without estate planning."
        
        response = f"Found {len(hnw_no_planning)} HNW clients without estate planning:\n\n"
        for c in sorted(hnw_no_planning, key=lambda x: x["portfolio"], reverse=True)[:20]:
            response += f"• {c['name']}: Portfolio £{c['portfolio']:,.0f}\n"
            response += f"  Will: {'Yes' if c['has_will'] else 'No'}, Trust: {'Yes' if c['has_trust'] else 'No'}\n\n"
        
        return response
    
    def _get_pension_clients_for_cashflow(self, query: str = "") -> str:
        """Get pension clients for cashflow modelling"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        pension_clients = []
        for client_id, client in data["clients"].items():
            products = client.get("products", [])
            has_pension = any("pension" in p.lower() or "SIPP" in p for p in products)
            if has_pension and not client.get("is_retired", False):
                pension_clients.append({
                    "name": client["name"],
                    "age": client["age"],
                    "retirement_age": client.get("retirement_age", 65),
                    "portfolio": client["portfolio_value_gbp"]
                })
        
        if not pension_clients:
            return "No pension clients found who might benefit from cashflow modelling."
        
        response = f"Found {len(pension_clients)} pension clients for cashflow modelling:\n\n"
        for c in pension_clients[:20]:
            response += f"• {c['name']}: Age {c['age']}, Retiring at {c['retirement_age']}, Portfolio: £{c['portfolio']:,.0f}\n"
        
        return response
    
    def _get_clients_investments_no_protection(self, query: str = "") -> str:
        """Find clients with investments but no protection"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        no_protection = []
        for client_id, client in data["clients"].items():
            has_investments = client.get("portfolio_value_gbp", 0) > 50000
            has_any_protection = (client.get("has_life_insurance", False) or 
                                 client.get("has_critical_illness", False) or 
                                 client.get("has_income_protection", False))
            
            if has_investments and not has_any_protection:
                no_protection.append({
                    "name": client["name"],
                    "portfolio": client["portfolio_value_gbp"],
                    "age": client["age"],
                    "dependents": client.get("num_dependents", 0)
                })
        
        if not no_protection:
            return "No clients found with investment portfolios but no protection cover."
        
        response = f"Found {len(no_protection)} clients with investments but no protection:\n\n"
        for c in sorted(no_protection, key=lambda x: x["portfolio"], reverse=True)[:20]:
            response += f"• {c['name']}: Portfolio £{c['portfolio']:,.0f}, Age {c['age']}, {c['dependents']} dependents\n"
        
        return response
    
    def _get_business_owners_no_exit_planning(self, query: str = "") -> str:
        """Get business owners without exit planning"""
        # This would require tracking exit planning discussions - using business owners as proxy
        business_owners = []
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        for client_id, client in data["clients"].items():
            if client.get("is_business_owner", False):
                # Check if exit planning mentioned in meetings
                has_exit_planning = False
                for meeting in data.get("meetings", []):
                    if meeting["client_id"] == client_id:
                        if "exit" in meeting.get("transcript", "").lower() or "succession" in meeting.get("transcript", "").lower():
                            has_exit_planning = True
                            break
                
                if not has_exit_planning:
                    business_owners.append({
                        "name": client["name"],
                        "age": client["age"],
                        "business_type": client.get("business_type", "Unknown")
                    })
        
        if not business_owners:
            return "No business owner clients found without exit planning discussions."
        
        response = f"Found {len(business_owners)} business owners without exit planning:\n\n"
        for c in business_owners[:20]:
            response += f"• {c['name']}: Age {c['age']}, {c['business_type']}\n"
        
        return response
    
    def _get_clients_birthdays_this_month(self, query: str = "") -> str:
        """Get clients with birthdays this month"""
        from datetime import datetime
        current_month = datetime.now().month
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        birthdays = []
        for client_id, client in data["clients"].items():
            dob = datetime.strptime(client["date_of_birth"], "%Y-%m-%d")
            if dob.month == current_month:
                birthdays.append({
                    "name": client["name"],
                    "birthday": dob.strftime("%B %d"),
                    "age": client["age"],
                    "turning_age": client["age"] + 1
                })
        
        if not birthdays:
            return "No clients have birthdays this month."
        
        response = f"Found {len(birthdays)} clients with birthdays this month:\n\n"
        for c in sorted(birthdays, key=lambda x: x["birthday"]):
            response += f"• {c['name']}: {c['birthday']} (turning {c['turning_age']})\n"
        
        return response
    
    # Compliance tools
    def _get_recommendations_for_client(self, query: str) -> str:
        """Get recommendations made to a client"""
        # Extract client name from query
        client_name = query.strip()
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        all_recommendations = []
        for meeting in data.get("meetings", []):
            if client_name.lower() in meeting["client_name"].lower():
                for rec in meeting.get("recommendations", []):
                    all_recommendations.append({
                        "client": meeting["client_name"],
                        "date": meeting["date"],
                        "recommendation": rec.get("recommendation", ""),
                        "rationale": rec.get("rationale", ""),
                        "status": rec.get("status", "")
                    })
        
        if not all_recommendations:
            return f"No recommendations found for {client_name}."
        
        response = f"Recommendations made to {client_name}:\n\n"
        for rec in all_recommendations:
            response += f"Date: {rec['date']}\n"
            response += f"Recommendation: {rec['recommendation']}\n"
            response += f"Rationale: {rec['rationale']}\n"
            response += f"Status: {rec['status']}\n\n"
        
        return response
    
    def _search_conversation_wording(self, query: str) -> str:
        """Search for exact wording in conversations"""
        results = self.vector_store.search(query, k=10)
        
        if not results:
            return f"No conversations found matching: {query}"
        
        response = f"Found {len(results)} conversations matching '{query}':\n\n"
        for i, doc in enumerate(results, 1):
            client_name = doc.metadata.get("client_name", "Unknown")
            date = doc.metadata.get("date", "Unknown")
            response += f"{i}. {client_name} - {date}\n"
            # Find the relevant excerpt
            content = doc.page_content
            query_lower = query.lower()
            if query_lower in content.lower():
                idx = content.lower().find(query_lower)
                start = max(0, idx - 100)
                end = min(len(content), idx + len(query) + 100)
                excerpt = content[start:end]
                response += f"   ...{excerpt}...\n\n"
            else:
                response += f"   {content[:200]}...\n\n"
        
        return response
    
    def _get_clients_recommended_platform(self, query: str) -> str:
        """Get clients where platform was recommended"""
        platform_name = query.strip()
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        clients_recommended = []
        for meeting in data.get("meetings", []):
            platforms = meeting.get("platforms_mentioned", [])
            if platform_name.lower() in [p.lower() for p in platforms]:
                # Find recommendation mentioning platform
                for rec in meeting.get("recommendations", []):
                    if platform_name.lower() in rec.get("recommendation", "").lower():
                        clients_recommended.append({
                            "client": meeting["client_name"],
                            "date": meeting["date"],
                            "recommendation": rec.get("recommendation", ""),
                            "rationale": rec.get("rationale", "")
                        })
                        break
        
        if not clients_recommended:
            return f"No clients found where {platform_name} was recommended."
        
        response = f"Clients where {platform_name} was recommended:\n\n"
        for c in clients_recommended:
            response += f"• {c['client']} ({c['date']}):\n"
            response += f"  {c['recommendation']}\n"
            response += f"  Reason: {c['rationale']}\n\n"
        
        return response
    
    def _get_conversations_about_topic(self, query: str) -> str:
        """Get conversations about a topic"""
        results = self.vector_store.search(query, k=15)
        
        if not results:
            return f"No conversations found about: {query}"
        
        response = f"Found {len(results)} conversations about '{query}':\n\n"
        for i, doc in enumerate(results, 1):
            client_name = doc.metadata.get("client_name", "Unknown")
            date = doc.metadata.get("date", "Unknown")
            response += f"{i}. {client_name} - {date}\n"
            response += f"   {doc.page_content[:300]}...\n\n"
        
        return response
    
    def _get_discussion_summary(self, query: str) -> str:
        """Get summary of discussions about a topic"""
        results = self.vector_store.search(query, k=20)
        
        if not results:
            return f"No discussions found about: {query}"
        
        # Group by client
        client_discussions = {}
        for doc in results:
            client_name = doc.metadata.get("client_name", "Unknown")
            if client_name not in client_discussions:
                client_discussions[client_name] = []
            client_discussions[client_name].append({
                "date": doc.metadata.get("date", "Unknown"),
                "content": doc.page_content[:200]
            })
        
        response = f"Summary of discussions about '{query}':\n\n"
        response += f"Total conversations: {len(results)}\n"
        response += f"Unique clients: {len(client_discussions)}\n\n"
        
        for client, discussions in list(client_discussions.items())[:10]:
            response += f"{client}:\n"
            for disc in discussions[:2]:  # Max 2 per client
                response += f"  • {disc['date']}: {disc['content']}...\n"
            response += "\n"
        
        return response
    
    def _get_documents_waiting(self, query: str = "") -> str:
        """Get documents waiting for"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        all_documents = []
        for meeting in data.get("meetings", []):
            for doc in meeting.get("documents_waiting", []):
                all_documents.append({
                    "client": meeting["client_name"],
                    "document_type": doc.get("document_type", ""),
                    "requested_date": doc.get("requested_date", ""),
                    "status": doc.get("status", "")
                })
        
        if not all_documents:
            return "No documents currently waiting for from clients."
        
        response = f"Documents waiting for ({len(all_documents)}):\n\n"
        for doc in all_documents[:20]:
            response += f"• {doc['client']}: {doc['document_type']} (requested: {doc['requested_date']})\n"
        
        return response
    
    def _get_promises_made(self, query: str = "") -> str:
        """Get promises made to clients"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        all_promises = []
        for meeting in data.get("meetings", []):
            for promise in meeting.get("promises", []):
                all_promises.append({
                    "client": meeting["client_name"],
                    "promise": promise.get("promise", ""),
                    "promised_date": promise.get("promised_date", ""),
                    "status": promise.get("status", "")
                })
        
        if not all_promises:
            return "No promises found in meeting records."
        
        response = f"Promises made to clients ({len(all_promises)}):\n\n"
        for p in sorted(all_promises, key=lambda x: x["promised_date"])[:20]:
            status_icon = "⚠️" if p["status"] == "overdue" else "✓"
            response += f"{status_icon} {p['client']}: {p['promise']} (promised: {p['promised_date']})\n"
        
        return response
    
    # Business analytics tools
    def _get_concerns_this_month(self, query: str = "") -> str:
        """Get concerns raised this month"""
        from datetime import datetime
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        concerns_this_month = []
        for meeting in data.get("meetings", []):
            meeting_date = datetime.strptime(meeting["date"], "%Y-%m-%d")
            if meeting_date.month == current_month and meeting_date.year == current_year:
                for concern in meeting.get("concerns_raised", []):
                    concerns_this_month.append({
                        "client": meeting["client_name"],
                        "date": meeting["date"],
                        "concern": concern
                    })
        
        if not concerns_this_month:
            return "No concerns raised in meetings this month."
        
        response = f"Concerns raised this month ({len(concerns_this_month)}):\n\n"
        # Group by concern type
        by_concern = {}
        for c in concerns_this_month:
            concern_type = c["concern"]
            if concern_type not in by_concern:
                by_concern[concern_type] = []
            by_concern[concern_type].append(c["client"])
        
        for concern_type, clients in by_concern.items():
            response += f"• {concern_type}: {len(clients)} clients\n"
            response += f"  Clients: {', '.join(clients[:5])}\n\n"
        
        return response
    
    def _get_service_usage_analysis(self, query: str = "") -> str:
        """Analyze service usage by high-value clients"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        # Define high-value threshold (top 20%)
        portfolios = [c["portfolio_value_gbp"] for c in data["clients"].values()]
        threshold = sorted(portfolios, reverse=True)[len(portfolios) // 5]
        
        high_value_clients = {cid: c for cid, c in data["clients"].items() 
                             if c["portfolio_value_gbp"] >= threshold}
        
        # Analyze products used
        product_usage = {}
        for cid, client in high_value_clients.items():
            for product in client.get("products", []):
                product_usage[product] = product_usage.get(product, 0) + 1
        
        if not product_usage:
            return "No service usage data available."
        
        response = f"Service usage by high-value clients (top 20%, {len(high_value_clients)} clients):\n\n"
        for product, count in sorted(product_usage.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(high_value_clients)) * 100
            response += f"• {product}: {count} clients ({pct:.1f}%)\n"
        
        return response
    
    def _get_conversion_rates(self, query: str = "") -> str:
        """Get conversion rates by referral source"""
        # Mock data - in real system this would come from CRM
        response = "Conversion rates by referral source:\n\n"
        response += "• Existing client referral: 65% (130/200 initial meetings)\n"
        response += "• Professional referral: 45% (45/100 initial meetings)\n"
        response += "• Website/Online: 25% (25/100 initial meetings)\n"
        response += "• Other: 30% (30/100 initial meetings)\n\n"
        response += "Note: This is mock data. In production, this would integrate with CRM."
        return response
    
    def _get_book_demographics(self, query: str = "") -> str:
        """Analyze book demographics"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        total = len(data["clients"])
        approaching_retirement = sum(1 for c in data["clients"].values() 
                                   if 60 <= c["age"] <= 65)
        retired = sum(1 for c in data["clients"].values() if c.get("is_retired", False))
        business_owners = sum(1 for c in data["clients"].values() 
                            if c.get("is_business_owner", False))
        hnw = sum(1 for c in data["clients"].values() 
                 if c.get("is_high_net_worth", False))
        
        response = f"Client Book Demographics ({total} total clients):\n\n"
        response += f"• Approaching retirement (60-65): {approaching_retirement} ({approaching_retirement/total*100:.1f}%)\n"
        response += f"• Retired: {retired} ({retired/total*100:.1f}%)\n"
        response += f"• Business owners: {business_owners} ({business_owners/total*100:.1f}%)\n"
        response += f"• High net worth: {hnw} ({hnw/total*100:.1f}%)\n"
        
        return response
    
    def _get_revenue_time_analysis(self, query: str = "") -> str:
        """Analyze revenue vs time"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        # Mock analysis - in production would use actual revenue and time data
        # Using portfolio value as proxy for revenue potential
        clients_with_data = []
        for cid, client in data["clients"].items():
            # Estimate time based on number of meetings and review status
            num_meetings = len([m for m in data.get("meetings", []) if m["client_id"] == cid])
            estimated_time = num_meetings * 1.5  # hours per year (rough estimate)
            portfolio = client["portfolio_value_gbp"]
            # Revenue proxy: assume 1% of portfolio as annual revenue
            revenue_proxy = portfolio * 0.01
            
            if estimated_time > 0:
                efficiency = revenue_proxy / estimated_time
                clients_with_data.append({
                    "name": client["name"],
                    "revenue": revenue_proxy,
                    "time": estimated_time,
                    "efficiency": efficiency
                })
        
        if not clients_with_data:
            return "No data available for revenue-time analysis."
        
        # Top 10 most efficient
        top_efficient = sorted(clients_with_data, key=lambda x: x["efficiency"], reverse=True)[:10]
        
        response = "Clients generating most revenue with least time:\n\n"
        for c in top_efficient:
            response += f"• {c['name']}: £{c['revenue']:,.0f}/year, {c['time']:.1f} hours (£{c['efficiency']:,.0f}/hour)\n"
        
        response += "\nNote: Revenue is estimated based on portfolio value. Time is estimated from meeting frequency."
        return response
    
    def _get_satisfied_client_patterns(self, query: str = "") -> str:
        """Analyze satisfied client patterns"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        # Mock: satisfied = long-term clients with multiple products and recent contact
        satisfied = []
        for cid, client in data["clients"].items():
            num_products = len(client.get("products", []))
            days_since_contact = client.get("days_since_contact", 365)
            num_meetings = len([m for m in data.get("meetings", []) if m["client_id"] == cid])
            
            if num_products >= 2 and days_since_contact < 180 and num_meetings >= 3:
                satisfied.append({
                    "name": client["name"],
                    "products": num_products,
                    "meetings": num_meetings,
                    "years_client": num_meetings / 2  # rough estimate
                })
        
        if not satisfied:
            return "No clear patterns identified for satisfied clients."
        
        response = f"Patterns from {len(satisfied)} satisfied long-term clients:\n\n"
        avg_products = sum(c["products"] for c in satisfied) / len(satisfied)
        avg_meetings = sum(c["meetings"] for c in satisfied) / len(satisfied)
        
        response += f"• Average products: {avg_products:.1f}\n"
        response += f"• Average meetings: {avg_meetings:.1f}\n"
        response += f"• Common products: Multiple (diversified portfolios)\n"
        response += f"• Regular contact: Yes (within 6 months)\n"
        
        return response
    
    def _get_recommendation_pushback(self, query: str = "") -> str:
        """Analyze recommendation pushback"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        pushback_by_type = {}
        total_by_type = {}
        
        for meeting in data.get("meetings", []):
            for rec in meeting.get("recommendations", []):
                rec_type = rec.get("type", "other")
                status = rec.get("status", "")
                
                total_by_type[rec_type] = total_by_type.get(rec_type, 0) + 1
                if status == "declined":
                    pushback_by_type[rec_type] = pushback_by_type.get(rec_type, 0) + 1
        
        if not pushback_by_type:
            return "No significant pushback patterns identified."
        
        response = "Recommendation pushback analysis:\n\n"
        for rec_type in sorted(pushback_by_type.keys()):
            declined = pushback_by_type[rec_type]
            total = total_by_type.get(rec_type, 1)
            pct = (declined / total) * 100
            response += f"• {rec_type.title()}: {declined}/{total} declined ({pct:.1f}%)\n"
        
        return response
    
    def _get_similar_circumstances_cases(self, query: str) -> str:
        """Find similar circumstances cases"""
        # Use search to find similar cases
        results = self.vector_store.search(query, k=10)
        
        if not results:
            return "No similar cases found. Try describing the circumstances more specifically."
        
        response = f"Found {len(results)} similar cases:\n\n"
        for i, doc in enumerate(results, 1):
            client_name = doc.metadata.get("client_name", "Unknown")
            response += f"{i}. {client_name}\n"
            response += f"   {doc.page_content[:250]}...\n\n"
        
        return response
    
    def _get_life_events_triggering_implementation(self, query: str = "") -> str:
        """Identify life events triggering implementation"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        # Analyze which life events correlate with accepted recommendations
        event_implementation = {}
        
        for cid, client in data["clients"].items():
            for event in client.get("life_events", []):
                event_type = event["type"]
                # Check if client has accepted recommendations
                for meeting in data.get("meetings", []):
                    if meeting["client_id"] == cid:
                        for rec in meeting.get("recommendations", []):
                            if rec.get("status") == "accepted":
                                if event_type not in event_implementation:
                                    event_implementation[event_type] = {"total": 0, "implemented": 0}
                                event_implementation[event_type]["total"] += 1
                                event_implementation[event_type]["implemented"] += 1
                                break
        
        if not event_implementation:
            return "No clear patterns identified for life events triggering implementation."
        
        response = "Life events triggering recommendation implementation:\n\n"
        for event_type, data in sorted(event_implementation.items(), 
                                       key=lambda x: x[1]["implemented"], reverse=True):
            pct = (data["implemented"] / data["total"]) * 100 if data["total"] > 0 else 0
            response += f"• {event_type}: {data['implemented']}/{data['total']} ({pct:.0f}%)\n"
        
        return response
    
    # Follow-up and actions tools
    def _draft_follow_up_email(self, query: str) -> str:
        """Draft follow-up email"""
        # Extract client name from query
        import re
        client_match = re.search(r'(\w+\s+\w+)', query)
        client_name = None
        
        # Try to find client name in query
        if client_match:
            potential_name = client_match.group(1)
            # Use search to find the client
            results = self.vector_store.search(query, k=5)
            if results:
                client_name = results[0].metadata.get("client_name", potential_name)
        
        # If no client found, try to extract from query more directly
        if not client_name:
            # Try searching for "yesterday's meeting" or similar
            results = self.vector_store.search(query, k=5)
            if results:
                client_name = results[0].metadata.get("client_name")
        
        if not client_name:
            return "Could not identify the client. Please specify the client name (e.g., 'Draft follow-up email for John Smith')."
        
        # Find the most recent meeting for this client
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        # Get all meetings for this client, sorted by date (most recent first)
        client_meetings = []
        for meeting in data.get("meetings", []):
            if client_name.lower() in meeting["client_name"].lower() or meeting["client_name"].lower() in client_name.lower():
                client_meetings.append(meeting)
        
        if not client_meetings:
            return f"Could not find any meetings for {client_name}. Please check the client name."
        
        # Sort by date (most recent first)
        from datetime import datetime
        client_meetings.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"), reverse=True)
        most_recent_meeting = client_meetings[0]
        
        meeting_date = most_recent_meeting["date"]
        actions = most_recent_meeting.get("action_items", [])
        meeting_type = most_recent_meeting.get("type", "meeting")
        
        # Format the date nicely
        try:
            date_obj = datetime.strptime(meeting_date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%B %d, %Y")
        except:
            formatted_date = meeting_date
        
        # Get first name
        first_name = client_name.split()[0] if client_name else "Client"
        
        # Draft the email
        response = f"Draft Follow-up Email\n"
        response += "=" * 60 + "\n\n"
        response += f"To: {client_name}\n"
        response += f"Subject: Follow-up from our {meeting_type} on {formatted_date}\n\n"
        response += f"Dear {first_name},\n\n"
        response += f"Thank you for taking the time to meet with me on {formatted_date}. I wanted to follow up on our discussion.\n\n"
        
        if actions:
            response += "Key Actions Agreed:\n"
            for action in actions[:5]:
                action_text = action.get('action', '')
                due_date = action.get('due_date', 'TBD')
                response += f"• {action_text} (Due: {due_date})\n"
            response += "\n"
        else:
            # If no specific actions, mention the meeting topics
            topics = most_recent_meeting.get("topics_discussed", [])
            if topics:
                response += "As discussed, we covered the following topics:\n"
                for topic in topics[:3]:
                    response += f"• {topic}\n"
                response += "\n"
        
        response += "Please don't hesitate to reach out if you have any questions or if you'd like to discuss anything further.\n\n"
        response += "Best regards,\n"
        response += "[Your Name]"
        
        return response
    
    def _get_waiting_on_clients(self, query: str = "") -> str:
        """Get clients waiting on"""
        # Combine documents waiting and pending actions
        waiting = []
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        for meeting in data.get("meetings", []):
            # Documents waiting
            for doc in meeting.get("documents_waiting", []):
                waiting.append({
                    "client": meeting["client_name"],
                    "item": doc.get("document_type", "Document"),
                    "type": "document",
                    "date": doc.get("requested_date", "")
                })
            
            # Pending actions requiring client input
            for action in meeting.get("action_items", []):
                if action.get("status") in ["pending", "in_progress"]:
                    if "client" in action.get("action", "").lower() or "you" in action.get("action", "").lower():
                        waiting.append({
                            "client": meeting["client_name"],
                            "item": action.get("action", ""),
                            "type": "action",
                            "date": action.get("due_date", "")
                        })
        
        if not waiting:
            return "No clients currently waiting on for information or decisions."
        
        response = f"Clients waiting on ({len(waiting)} items):\n\n"
        for w in waiting[:20]:
            response += f"• {w['client']}: {w['item']} ({w['type']}, {w['date']})\n"
        
        return response
    
    def _get_open_action_items(self, query: str = "") -> str:
        """Get all open action items"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        open_actions = []
        for meeting in data.get("meetings", []):
            for action in meeting.get("action_items", []):
                if action.get("status") != "completed":
                    open_actions.append({
                        "client": meeting["client_name"],
                        "action": action.get("action", ""),
                        "due_date": action.get("due_date", ""),
                        "status": action.get("status", "pending"),
                        "meeting_date": meeting["date"]
                    })
        
        if not open_actions:
            return "No open action items across client base."
        
        response = f"Open Action Items ({len(open_actions)}):\n\n"
        for a in sorted(open_actions, key=lambda x: x["due_date"])[:30]:
            response += f"• {a['client']}: {a['action']}\n"
            response += f"  Due: {a['due_date']}, Status: {a['status']}\n\n"
        
        return response
    
    def _get_overdue_follow_ups(self, query: str = "") -> str:
        """Get overdue follow-ups"""
        overdue = self.compliance_tracker.get_overdue_follow_ups()
        
        if not overdue:
            return "No overdue follow-up commitments."
        
        response = f"Overdue Follow-ups ({len(overdue)}):\n\n"
        for f in overdue[:20]:
            response += f"• {f['client_name']}: {f['follow_up_type']}\n"
            response += f"  Scheduled: {f['scheduled_date']}, {f['days_overdue']} days overdue\n\n"
        
        return response
    
    # Scenario modeling and advanced analysis tools
    def _analyze_interest_rate_impact(self, query: str) -> str:
        """Analyze which clients would be impacted if interest rates drop to specified level"""
        target_rate = 3.0
        import re
        rate_match = re.search(r'(\d+(?:\.\d+)?)%', query)
        if rate_match:
            target_rate = float(rate_match.group(1))
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        impacted = []
        for cid, client in data["clients"].items():
            portfolio = client.get("portfolio", {})
            fixed_income_pct = portfolio.get("fixed_income", 0)
            cash_pct = portfolio.get("cash", 0)
            
            if fixed_income_pct > 20 or cash_pct > 30:
                impact = "High" if (fixed_income_pct + cash_pct) > 50 else "Medium"
                impacted.append({
                    "name": client["name"],
                    "fixed_income": fixed_income_pct,
                    "cash": cash_pct,
                    "total_exposure": fixed_income_pct + cash_pct,
                    "impact": impact
                })
        
        if not impacted:
            return f"No clients significantly impacted by interest rate drop to {target_rate}%."
        
        response = f"Interest Rate Impact Analysis (Rate: {target_rate}%)\n"
        response += "=" * 60 + "\n\n"
        response += f"Found {len(impacted)} clients with significant fixed income/cash exposure:\n\n"
        
        for client in sorted(impacted, key=lambda x: x["total_exposure"], reverse=True)[:15]:
            response += f"• {client['name']}: {client['impact']} impact\n"
            response += f"  Fixed Income: {client['fixed_income']}%, Cash: {client['cash']}%\n"
            response += f"  Total Exposure: {client['total_exposure']}%\n\n"
        
        return response
    
    def _analyze_market_correction_exposure(self, query: str) -> str:
        """Analyze which clients are most exposed to a market correction"""
        correction_pct = 20.0
        import re
        correction_match = re.search(r'(\d+(?:\.\d+)?)%', query)
        if correction_match:
            correction_pct = float(correction_match.group(1))
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        exposed = []
        for cid, client in data["clients"].items():
            portfolio = client.get("portfolio", {})
            equity_pct = portfolio.get("equities", 0)
            risk_profile = client.get("risk_profile", "Moderate")
            
            if equity_pct > 50:
                exposure_score = equity_pct * (1.5 if risk_profile == "High" else 1.0)
                exposed.append({
                    "name": client["name"],
                    "equity_pct": equity_pct,
                    "risk_profile": risk_profile,
                    "exposure_score": exposure_score,
                    "estimated_loss": (equity_pct * correction_pct) / 100
                })
        
        if not exposed:
            return f"No clients significantly exposed to {correction_pct}% market correction."
        
        response = f"Market Correction Exposure Analysis ({correction_pct}% correction)\n"
        response += "=" * 60 + "\n\n"
        response += f"Found {len(exposed)} clients with high equity exposure:\n\n"
        
        for client in sorted(exposed, key=lambda x: x["exposure_score"], reverse=True)[:15]:
            response += f"• {client['name']}: {client['risk_profile']} risk profile\n"
            response += f"  Equity Allocation: {client['equity_pct']}%\n"
            response += f"  Estimated Impact: ~{client['estimated_loss']:.1f}% of portfolio\n\n"
        
        return response
    
    def _model_retirement_scenario(self, query: str) -> str:
        """Model retirement scenario changes"""
        import re
        client_match = re.search(r'(\w+)\s+(?:retires?|retirement)', query, re.IGNORECASE)
        
        if not client_match:
            return "Please specify the client name (e.g., 'If Roshan retires next year...')"
        
        client_name = client_match.group(1)
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        client = None
        for cid, c in data["clients"].items():
            if client_name.lower() in c["name"].lower():
                client = c
                break
        
        if not client:
            return f"Client '{client_name}' not found."
        
        response = f"Retirement Scenario: {client['name']}\n"
        response += "=" * 60 + "\n\n"
        response += "Analysis: Review cashflow projections and adjust plan accordingly.\n"
        return response
    
    def _model_long_term_care_scenario(self, query: str) -> str:
        """Model long-term care scenario"""
        import re
        family_match = re.search(r"(\w+)'?s?\s+family", query, re.IGNORECASE)
        
        if not family_match:
            return "Please specify the family name (e.g., 'Gurung's family...')"
        
        family_name = family_match.group(1)
        
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        family_members = []
        for cid, client in data["clients"].items():
            if family_name.lower() in client["name"].lower():
                family_members.append(client)
        
        if not family_members:
            return f"Family '{family_name}' not found."
        
        response = f"Long-Term Care Scenario: {family_name} Family\n"
        response += "=" * 60 + "\n\n"
        response += f"Estimated Annual Cost: £50,000\n"
        response += "Recommendations: Review care insurance and estate planning.\n"
        return response
    
    def _get_business_owners_rd_tax_credit(self, query: str = "") -> str:
        """Get business owners for R&D tax credit opportunities"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        business_owners = []
        for cid, client in data["clients"].items():
            if client.get("occupation_type") == "Business Owner":
                business_owners.append({
                    "name": client["name"],
                    "business_type": client.get("business_type", "Not specified")
                })
        
        if not business_owners:
            return "No business owner clients found."
        
        response = "Business Owners - R&D Tax Credit Opportunities\n"
        response += "=" * 60 + "\n\n"
        for owner in business_owners[:15]:
            response += f"• {owner['name']}: {owner['business_type']}\n"
        
        response += "\nRecent Changes: Enhanced rate 130% deduction, 14.5% credit for loss-makers.\n"
        return response
    
    def chat(self, message: str, conversation_history: list = None) -> str:
        """Main chat interface with improved error handling, logging, and conversation history support"""
        logger.info(f"=== CHAT START: {message[:100]} ===")
        
        # Build messages list with conversation history
        messages = []
        if conversation_history:
            logger.info(f"Loading {len(conversation_history)} previous messages for context")
            for msg in conversation_history:
                # Handle different message formats from frontend
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", msg.get("message", ""))
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                elif isinstance(msg, str):
                    # Assume user message if just a string
                    messages.append(HumanMessage(content=msg))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Always use agentic workflow - LLM decides which tools to use
        initial_state = {
            "messages": messages,  # Use messages with history
            "current_query": message,
            "context": {"conversation_history": len(conversation_history) if conversation_history else 0},
            "compliance_data": {}
        }
        
        try:
            logger.info("Invoking LangGraph agentic workflow - LLM will decide which tools to use...")
            result = self.graph.invoke(initial_state)
            logger.info(f"Graph execution complete. Messages in result: {len(result.get('messages', []))}")
            
            # Log all messages for debugging
            for idx, msg in enumerate(result.get("messages", [])):
                msg_type = type(msg).__name__
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, 'tool_calls', None)
                    logger.debug(f"Message {idx}: {msg_type}, has_tool_calls: {bool(tool_calls)}, content_length: {len(msg.content) if msg.content else 0}")
                else:
                    logger.debug(f"Message {idx}: {msg_type}")
            
            if result and "messages" in result:
                # Extract the final AI message (skip tool messages)
                logger.info("Extracting final response from messages...")
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        tool_calls = getattr(msg, 'tool_calls', None)
                        if not tool_calls or len(tool_calls) == 0:
                            content = msg.content
                            if content:
                                logger.info(f"Found final response, length: {len(content)}")
                                logger.info(f"=== CHAT END: Response found ===")
                                return content
                            else:
                                logger.warning("AIMessage has no content")
            
            # Fallback - get last message content
            if result.get("messages"):
                last_msg = result["messages"][-1]
                logger.info(f"Using last message as fallback: {type(last_msg).__name__}")
                if isinstance(last_msg, AIMessage):
                    response = last_msg.content or "I'm processing your request..."
                    logger.info(f"=== CHAT END: Fallback response ===")
                    return response
                else:
                    logger.warning(f"Last message is not AIMessage: {type(last_msg).__name__}")
                    logger.info(f"=== CHAT END: No valid response ===")
                    return str(last_msg)
            
            logger.warning("No result or messages found")
            logger.info(f"=== CHAT END: No result ===")
            return "I'm processing your request..."
            
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Error in chat method: {e}", exc_info=True)
            
            # Check for rate limit or quota errors - provide helpful error message
            if any(keyword in error_str for keyword in ["rate_limit", "quota", "429", "insufficient_quota", "rate limit", "too many requests"]):
                self.fallback_mode = True
                self.rate_limit_errors += 1
                logger.warning("Rate limit/quota error detected")
                logger.info(f"=== CHAT END: Rate limit error ===")
                return "I'm experiencing API rate limits. Please try again in a moment. The system uses an agentic workflow where the AI intelligently selects the right tools for your query."
            
            # Try to provide helpful response based on error
            if "tool" in error_str and "validation" in error_str:
                logger.warning("Tool validation error - this should not happen in agentic workflow")
                logger.info(f"=== CHAT END: Tool validation error ===")
                return "There was an error with tool validation. Please rephrase your question."
            elif "413" in error_str or "too large" in error_str or "tpm" in error_str:
                logger.warning("Request too large")
                logger.info(f"=== CHAT END: Request too large ===")
                return "Your query is too large. Please try a shorter, more focused question."
            else:
                logger.error(f"Unexpected error: {e}")
                logger.info(f"=== CHAT END: Error ===")
                return f"I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def get_status(self) -> dict:
        """Get current status including rate limit and fallback mode"""
        time_diff = (datetime.now() - self.rate_limit_window_start).total_seconds()
        tokens_remaining = max(0, self.rate_limit_threshold - self.current_token_count)
        
        return {
            "fallback_mode": self.fallback_mode,
            "current_tokens": self.current_token_count,
            "token_limit": self.rate_limit_threshold,
            "tokens_remaining": tokens_remaining,
            "rate_limit_errors": self.rate_limit_errors,
            "window_reset_in": max(0, 60 - time_diff),
            "status": "fallback" if self.fallback_mode else "normal"
        }
    
    def _fallback_direct_tool_execution(self, query: str) -> str:
        """Fallback: directly execute tools based on query content"""
        logger.info(f"Fallback direct tool execution for: {query[:100]}")
        query_lower = query.lower()
        
        # Map queries to tools - expanded matching for better coverage
        if "daily briefing" in query_lower or "what needs attention" in query_lower or "briefing" in query_lower or "needs attention" in query_lower:
            logger.info("Executing: get_daily_briefing")
            return self._get_daily_briefing("")
        elif "review" in query_lower:
            if ("12" in query_lower or "twelve" in query_lower) and ("month" in query_lower or "overdue" in query_lower):
                logger.info("Executing: get_clients_reviews_overdue_12_months")
                return self._get_clients_reviews_overdue_12_months("")
            elif "due" in query_lower or "overdue" in query_lower or "annual review" in query_lower:
                logger.info("Executing: get_reviews_due")
                return self._get_reviews_due("")
        elif "isa" in query_lower and "allowance" in query_lower:
            logger.info("Executing: get_clients_with_isa_allowance")
            return self._get_clients_with_isa_allowance("")
        elif "annual allowance" in query_lower and "pension" not in query_lower:
            logger.info("Executing: get_clients_with_annual_allowance")
            return self._get_clients_with_annual_allowance("")
        elif "cash excess" in query_lower or ("cash" in query_lower and "excess" in query_lower and "6" in query_lower):
            logger.info("Executing: get_clients_with_cash_excess")
            return self._get_clients_with_cash_excess("")
        elif "equity" in query_lower and ("underweight" in query_lower or "allocation" in query_lower):
            logger.info("Executing: get_clients_underweight_equities")
            return self._get_clients_underweight_equities("")
        elif "business owner" in query_lower:
            logger.info("Executing: get_business_owners")
            return self._get_business_owners("")
        elif "birthday" in query_lower or "birthdays" in query_lower:
            logger.info("Executing: get_clients_birthdays_this_month")
            return self._get_clients_birthdays_this_month("")
        elif "document" in query_lower and ("waiting" in query_lower or "outstanding" in query_lower):
            logger.info("Executing: get_documents_waiting")
            return self._get_documents_waiting("")
        elif "action" in query_lower and ("item" in query_lower or "open" in query_lower):
            logger.info("Executing: get_open_action_items")
            return self._get_open_action_items("")
        elif "protection gap" in query_lower:
            logger.info("Executing: get_clients_with_protection_gaps")
            return self._get_clients_with_protection_gaps("")
        elif "milestone" in query_lower:
            logger.info("Executing: get_upcoming_milestones")
            return self._get_upcoming_milestones("")
        elif "life event" in query_lower or ("life" in query_lower and "event" in query_lower):
            logger.info("Executing: get_life_events")
            return self._get_life_events("")
        elif "concern" in query_lower and ("unresolved" in query_lower or "client" in query_lower):
            logger.info("Executing: get_unresolved_concerns")
            return self._get_unresolved_concerns("")
        else:
            # Try search as last resort
            logger.info("No direct match - trying search_clients")
            return self._search_clients(query)
    
    # ==================== PROACTIVE INTELLIGENCE FEATURES ====================
    
    def _get_priority_scored_opportunities(self, query: str = "") -> str:
        """Get opportunities ranked by priority score (impact, urgency, value)"""
        opportunities = []
        
        # Get all different opportunity types
        reviews_due = self.compliance_tracker.get_reviews_due()
        contact_gaps = self.compliance_tracker.get_contact_gaps()
        milestones = self.compliance_tracker.get_upcoming_milestones()
        life_events = self.compliance_tracker.get_life_events_requiring_attention()
        
        # Load client data for value calculation
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        # Score reviews due
        for review in reviews_due[:30]:
            client_id = review.get('client_id')
            client = data["clients"].get(client_id, {})
            portfolio_value = client.get("portfolio_value_gbp", 0)
            
            score = self._calculate_priority_score(
                urgency=review.get('days_overdue', 0),
                client_value=portfolio_value,
                regulatory_risk=1.0 if review.get('days_overdue', 0) > 30 else 0.5,
                time_required=30
            )
            opportunities.append({
                'type': 'Review Due',
                'client': review['client_name'],
                'priority_score': score,
                'urgency': review.get('days_overdue', 0),
                'details': f"{review.get('days_overdue', 0)} days overdue",
                'action': 'Schedule annual review'
            })
        
        # Score contact gaps
        for gap in contact_gaps[:20]:
            client_id = gap.get('client_id')
            client = data["clients"].get(client_id, {})
            portfolio_value = client.get("portfolio_value_gbp", 0)
            
            score = self._calculate_priority_score(
                urgency=gap.get('days_since_contact', 0) / 10,
                client_value=portfolio_value,
                regulatory_risk=0.3,
                time_required=15
            )
            opportunities.append({
                'type': 'Contact Gap',
                'client': gap['client_name'],
                'priority_score': score,
                'urgency': gap.get('days_since_contact', 0),
                'details': f"{gap.get('days_since_contact', 0)} days since contact",
                'action': 'Reach out to reconnect'
            })
        
        # Score milestones
        for milestone in milestones[:15]:
            client_id = milestone.get('client_id')
            client = data["clients"].get(client_id, {})
            portfolio_value = client.get("portfolio_value_gbp", 0)
            days_until = milestone.get('days_until', 999)
            
            score = self._calculate_priority_score(
                urgency=max(0, 90 - days_until),  # Higher urgency as deadline approaches
                client_value=portfolio_value,
                regulatory_risk=0.2,
                time_required=45
            )
            opportunities.append({
                'type': 'Milestone Birthday',
                'client': milestone['client_name'],
                'priority_score': score,
                'urgency': days_until,
                'details': f"Turning {milestone.get('turning_age', 0)} in {days_until} days",
                'action': 'Plan milestone birthday strategy'
            })
        
        # Score ISA/Annual Allowance opportunities
        isa_result = self._get_clients_with_isa_allowance("")
        if "clients" in isa_result.lower():
            # Parse and score ISA opportunities
            for line in isa_result.split('\n')[1:6]:  # Top 5
                if '•' in line or '-' in line:
                    parts = line.replace('•', '').replace('-', '').strip().split('£')
                    if len(parts) > 1:
                        client_name = parts[0].strip()
                        allowance = float(parts[1].split()[0].replace(',', ''))
                        client = next((c for c in data["clients"].values() if c['name'] == client_name), {})
                        portfolio_value = client.get("portfolio_value_gbp", 0)
                        
                        score = self._calculate_priority_score(
                            urgency=30,  # Tax year deadline approaching
                            client_value=portfolio_value,
                            regulatory_risk=0.1,
                            time_required=20
                        )
                        opportunities.append({
                            'type': 'ISA Allowance',
                            'client': client_name,
                            'priority_score': score,
                            'urgency': 30,
                            'details': f"£{allowance:,.0f} allowance remaining",
                            'action': 'Discuss ISA contribution'
                        })
        
        # Sort by priority score
        opportunities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Format output
        output = "## 🎯 Top Priority Opportunities (Ranked by Impact)\n\n"
        output += "**Score = (Urgency × Client Value × Risk) / Time Required**\n\n"
        
        for i, opp in enumerate(opportunities[:15], 1):
            urgency_emoji = "🔴" if opp['urgency'] > 60 else "🟡" if opp['urgency'] > 30 else "🟢"
            output += f"{i}. **{opp['client']}** - {opp['type']}\n"
            output += f"   Priority Score: {opp['priority_score']:.1f} {urgency_emoji}\n"
            output += f"   {opp['details']}\n"
            output += f"   → Action: {opp['action']}\n\n"
        
        return output
    
    def _calculate_priority_score(self, urgency, client_value, regulatory_risk, time_required):
        """Calculate priority score (0-100)"""
        # Normalize urgency (0-10 scale)
        urgency_score = min(urgency / 10, 10)
        
        # Normalize client value (0-10 scale, assuming max portfolio ~2M)
        value_score = min(client_value / 200000, 10)
        
        # Risk multiplier (1.0 to 1.5)
        risk_multiplier = 1 + (regulatory_risk * 0.5)
        
        # Time efficiency (less time = higher multiplier)
        time_efficiency = max(0.5, 1.5 - (time_required / 120))
        
        # Final score
        score = (urgency_score * 3 + value_score * 2) * risk_multiplier * time_efficiency
        return round(score, 1)
    
    def _get_relationship_health_dashboard(self, query: str = "") -> str:
        """Get relationship health for all clients with risk analysis"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        health_scores = []
        
        for client_id, client in data["clients"].items():
            # Calculate engagement score (0-100)
            days_since_contact = client.get("days_since_contact", 365)
            engagement = max(0, 100 - (days_since_contact / 3.65))
            
            # Calculate risk factors
            risk_factors = []
            risk_count = 0
            
            if days_since_contact > 180:
                risk_factors.append("Long contact gap")
                risk_count += 1
            if client.get("days_since_review", 0) > 450:
                risk_factors.append("Overdue review")
                risk_count += 1
            if any(c.get("status") == "active" for c in client.get("concerns", [])):
                risk_factors.append("Unresolved concerns")
                risk_count += 1
            if client.get("has_protection_gap", False):
                risk_factors.append("Protection gap")
                risk_count += 1
            
            # Determine risk level
            if risk_count >= 2:
                risk_level = "High"
            elif risk_count == 1:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Calculate relationship momentum
            momentum = "Stable"
            if days_since_contact < 60 and client.get("days_since_review", 0) < 365:
                momentum = "Improving"
            elif days_since_contact > 180 or client.get("days_since_review", 0) > 450:
                momentum = "Declining"
            
            # Calculate value at risk
            portfolio_value = client.get("portfolio_value_gbp", 0)
            value_at_risk = portfolio_value * (risk_count * 0.15) if risk_count > 0 else 0
            
            health_scores.append({
                'client_name': client['name'],
                'client_id': client_id,
                'engagement_score': round(engagement, 1),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'risk_count': risk_count,
                'momentum': momentum,
                'value_at_risk': round(value_at_risk, 2),
                'portfolio_value': portfolio_value
            })
        
        # Sort by risk level and value at risk
        health_scores.sort(key=lambda x: (
            {'High': 3, 'Medium': 2, 'Low': 1}[x['risk_level']],
            -x['value_at_risk']
        ), reverse=True)
        
        # Format output
        output = "## 💚 Relationship Health Dashboard\n\n"
        
        high_risk = [h for h in health_scores if h['risk_level'] == 'High']
        medium_risk = [h for h in health_scores if h['risk_level'] == 'Medium']
        low_risk = [h for h in health_scores if h['risk_level'] == 'Low']
        
        output += f"### 🔴 High Risk Clients ({len(high_risk)} - Need Immediate Attention)\n\n"
        for client in high_risk[:10]:
            output += f"**{client['client_name']}**\n"
            output += f"- Engagement Score: {client['engagement_score']}/100\n"
            output += f"- Risk Factors: {', '.join(client['risk_factors'])}\n"
            output += f"- Value at Risk: £{client['value_at_risk']:,.0f}\n"
            output += f"- Momentum: {client['momentum']}\n"
            output += f"- Portfolio Value: £{client['portfolio_value']:,.0f}\n\n"
        
        if medium_risk:
            output += f"\n### 🟡 Medium Risk Clients ({len(medium_risk)})\n\n"
            for client in medium_risk[:5]:
                output += f"**{client['client_name']}** - {', '.join(client['risk_factors'])}\n"
                output += f"Engagement: {client['engagement_score']}/100 | Value at Risk: £{client['value_at_risk']:,.0f}\n\n"
        
        output += f"\n### 📊 Summary\n"
        output += f"- Total Clients: {len(health_scores)}\n"
        output += f"- High Risk: {len(high_risk)} ({len(high_risk)/len(health_scores)*100:.1f}%)\n"
        output += f"- Medium Risk: {len(medium_risk)} ({len(medium_risk)/len(health_scores)*100:.1f}%)\n"
        output += f"- Low Risk: {len(low_risk)} ({len(low_risk)/len(health_scores)*100:.1f}%)\n"
        output += f"- Total Value at Risk: £{sum(h['value_at_risk'] for h in health_scores):,.0f}\n"
        
        return output
    
    def _get_opportunity_window_alerts(self, query: str = "") -> str:
        """Get time-sensitive opportunities with countdown timers"""
        from datetime import datetime, timedelta
        
        alerts = []
        today = datetime.now()
        
        # Calculate tax year end (UK: April 5)
        tax_year_end = datetime(today.year, 4, 5)
        if today > tax_year_end:
            tax_year_end = datetime(today.year + 1, 4, 5)
        
        days_to_tax_year_end = (tax_year_end - today).days
        
        # ISA Allowance opportunities
        isa_clients = self._get_clients_with_isa_allowance("")
        if isa_clients and "clients" in isa_clients.lower():
            client_count = len([l for l in isa_clients.split('\n') if '•' in l or '-' in l])
            alerts.append({
                'type': 'ISA Allowance',
                'deadline': tax_year_end.strftime('%Y-%m-%d'),
                'days_remaining': days_to_tax_year_end,
                'clients_affected': client_count,
                'urgency': 'High' if days_to_tax_year_end < 30 else 'Medium' if days_to_tax_year_end < 90 else 'Low',
                'action': 'Contact clients to use remaining ISA allowance before tax year end',
                'value': 'Tax-free savings opportunity'
            })
        
        # Annual Allowance opportunities
        annual_clients = self._get_clients_with_annual_allowance("")
        if annual_clients and "clients" in annual_clients.lower():
            client_count = len([l for l in annual_clients.split('\n') if '•' in l or '-' in l])
            alerts.append({
                'type': 'Pension Annual Allowance',
                'deadline': tax_year_end.strftime('%Y-%m-%d'),
                'days_remaining': days_to_tax_year_end,
                'clients_affected': client_count,
                'urgency': 'High' if days_to_tax_year_end < 30 else 'Medium' if days_to_tax_year_end < 90 else 'Low',
                'action': 'Discuss pension contributions before tax year end',
                'value': 'Tax relief opportunity'
            })
        
        # Milestone birthdays (next 90 days)
        milestones = self.compliance_tracker.get_upcoming_milestones()
        upcoming_milestones = [m for m in milestones if m.get('days_until', 999) <= 90]
        if upcoming_milestones:
            min_days = min([m.get('days_until', 0) for m in upcoming_milestones])
            alerts.append({
                'type': 'Milestone Birthdays',
                'deadline': 'Next 90 days',
                'days_remaining': min_days,
                'clients_affected': len(upcoming_milestones),
                'urgency': 'High' if min_days < 30 else 'Medium',
                'action': 'Plan for milestone birthdays (pension access, tax planning)',
                'value': 'Life stage planning opportunity'
            })
        
        # Sort by urgency and days remaining
        urgency_order = {'High': 3, 'Medium': 2, 'Low': 1}
        alerts.sort(key=lambda x: (urgency_order[x['urgency']], x['days_remaining']))
        
        # Format output
        output = "## ⏰ Opportunity Window Alerts\n\n"
        output += "**Time-sensitive opportunities that require action**\n\n"
        
        for alert in alerts:
            urgency_emoji = "🔴" if alert['urgency'] == 'High' else "🟡" if alert['urgency'] == 'Medium' else "🟢"
            output += f"{urgency_emoji} **{alert['type']}**\n"
            output += f"- Deadline: {alert['deadline']} ({alert['days_remaining']} days remaining)\n"
            output += f"- Clients Affected: {alert['clients_affected']}\n"
            output += f"- Value: {alert['value']}\n"
            output += f"- Action: {alert['action']}\n\n"
        
        if not alerts:
            output += "No urgent opportunity windows closing soon.\n"
        
        return output
    
    def _get_predictive_risk_detection(self, query: str = "") -> str:
        """Identify potential problems before they become issues"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        risks = {
            'compliance': [],
            'relationship': [],
            'financial': [],
            'opportunity': []
        }
        
        for client_id, client in data["clients"].items():
            client_name = client['name']
            portfolio_value = client.get("portfolio_value_gbp", 0)
            
            # Compliance risks
            if client.get("days_since_review", 0) > 400:
                risks['compliance'].append({
                    'client': client_name,
                    'risk': 'Review overdue',
                    'severity': 'High' if client.get("days_since_review", 0) > 450 else 'Medium',
                    'days': client.get("days_since_review", 0),
                    'impact': 'FCA compliance issue'
                })
            
            # Relationship risks
            if client.get("days_since_contact", 0) > 180:
                risks['relationship'].append({
                    'client': client_name,
                    'risk': 'Contact gap',
                    'severity': 'High' if client.get("days_since_contact", 0) > 270 else 'Medium',
                    'days': client.get("days_since_contact", 0),
                    'impact': 'Relationship deterioration risk'
                })
            
            # Financial risks
            if client.get("withdrawal_rate_percent", 0) > 4 and client.get("is_retired", False):
                risks['financial'].append({
                    'client': client_name,
                    'risk': 'High withdrawal rate',
                    'severity': 'High',
                    'rate': client.get("withdrawal_rate_percent", 0),
                    'impact': 'Portfolio sustainability risk'
                })
            
            if client.get("has_protection_gap", False) and client.get("num_dependents", 0) > 0:
                risks['financial'].append({
                    'client': client_name,
                    'risk': 'Protection gap',
                    'severity': 'Medium',
                    'dependents': client.get("num_dependents", 0),
                    'impact': 'Family financial security risk'
                })
            
            # Opportunity risks (missing deadlines)
            if client.get("isa_allowance_available", 0) > 5000:
                from datetime import datetime
                today = datetime.now()
                tax_year_end = datetime(today.year, 4, 5)
                if today > tax_year_end:
                    tax_year_end = datetime(today.year + 1, 4, 5)
                days_remaining = (tax_year_end - today).days
                
                if days_remaining < 60:
                    risks['opportunity'].append({
                        'client': client_name,
                        'risk': 'ISA deadline approaching',
                        'severity': 'Medium',
                        'allowance': client.get("isa_allowance_available", 0),
                        'days': days_remaining,
                        'impact': 'Missing tax-free savings opportunity'
                    })
        
        # Format output
        output = "## 🚨 Predictive Risk Detection\n\n"
        output += "**Identifying problems before they become critical**\n\n"
        
        total_risks = sum(len(r) for r in risks.values())
        
        if total_risks == 0:
            output += "✅ No significant risks detected. Your client book is in good shape!\n"
            return output
        
        # Compliance risks
        if risks['compliance']:
            output += f"### 🔴 Compliance Risks ({len(risks['compliance'])})\n\n"
            for risk in risks['compliance'][:10]:
                output += f"**{risk['client']}** - {risk['risk']}\n"
                output += f"- Severity: {risk['severity']} | {risk['days']} days\n"
                output += f"- Impact: {risk['impact']}\n\n"
        
        # Relationship risks
        if risks['relationship']:
            output += f"### 💔 Relationship Risks ({len(risks['relationship'])})\n\n"
            for risk in risks['relationship'][:10]:
                output += f"**{risk['client']}** - {risk['risk']}\n"
                output += f"- Severity: {risk['severity']} | {risk['days']} days since contact\n"
                output += f"- Impact: {risk['impact']}\n\n"
        
        # Financial risks
        if risks['financial']:
            output += f"### 💰 Financial Risks ({len(risks['financial'])})\n\n"
            for risk in risks['financial'][:10]:
                output += f"**{risk['client']}** - {risk['risk']}\n"
                if 'rate' in risk:
                    output += f"- Withdrawal Rate: {risk['rate']}%\n"
                if 'dependents' in risk:
                    output += f"- Dependents: {risk['dependents']}\n"
                output += f"- Impact: {risk['impact']}\n\n"
        
        # Opportunity risks
        if risks['opportunity']:
            output += f"### ⏰ Opportunity Risks ({len(risks['opportunity'])})\n\n"
            output += "**Missing time-sensitive opportunities**\n\n"
            for risk in risks['opportunity'][:10]:
                output += f"**{risk['client']}** - {risk['risk']}\n"
                output += f"- Allowance: £{risk.get('allowance', 0):,.0f} | {risk.get('days', 0)} days remaining\n"
                output += f"- Impact: {risk['impact']}\n\n"
        
        output += f"\n### 📊 Risk Summary\n"
        output += f"- Total Risks Detected: {total_risks}\n"
        output += f"- High Severity: {sum(1 for r in risks.values() for item in r if item.get('severity') == 'High')}\n"
        output += f"- Medium Severity: {sum(1 for r in risks.values() for item in r if item.get('severity') == 'Medium')}\n"
        
        return output
    
    def _get_contextual_morning_briefing(self, query: str = "") -> str:
        """Get personalized contextual morning briefing"""
        from datetime import datetime
        
        today = datetime.now()
        day_name = today.strftime('%A')
        
        # Get standard briefing
        briefing = self.compliance_tracker.get_daily_briefing()
        
        # Get priority opportunities
        priority_opps = self._get_priority_scored_opportunities("")
        
        # Get opportunity alerts
        alerts = self._get_opportunity_window_alerts("")
        
        # Get top risks
        risks = self._get_predictive_risk_detection("")
        
        # Format contextual briefing
        output = f"## ☕ Good {day_name} Morning Briefing\n\n"
        output += f"**{today.strftime('%B %d, %Y')}**\n\n"
        
        output += "### 🎯 Your Top 3 Priorities Today\n\n"
        # Extract top 3 from priority opportunities
        lines = priority_opps.split('\n')
        count = 0
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.')) and '**' in line:
                output += line + '\n'
                count += 1
                if count >= 3:
                    break
        output += "\n"
        
        output += "### ⚠️ Critical Items Requiring Attention\n\n"
        summary = briefing["summary"]
        if summary['total_reviews_due'] > 0:
            output += f"🔴 **{summary['total_reviews_due']} Reviews Overdue** - Compliance risk\n"
        if summary['total_contact_gaps'] > 10:
            output += f"🟡 **{summary['total_contact_gaps']} Contact Gaps** - Relationship risk\n"
        if summary['total_overdue_actions'] > 0:
            output += f"🟠 **{summary['total_overdue_actions']} Overdue Actions** - Follow-up needed\n"
        output += "\n"
        
        output += "### ⏰ Time-Sensitive Opportunities\n\n"
        # Extract first alert
        alert_lines = alerts.split('\n')
        for i, line in enumerate(alert_lines[:10]):
            if '🔴' in line or '🟡' in line:
                output += line + '\n'
                # Get next few lines for context
                for j in range(1, 5):
                    if i + j < len(alert_lines):
                        output += alert_lines[i + j] + '\n'
                break
        output += "\n"
        
        output += "### 📊 Quick Stats\n\n"
        output += f"- Reviews Due: {summary['total_reviews_due']}\n"
        output += f"- Contact Gaps: {summary['total_contact_gaps']}\n"
        output += f"- Milestones: {summary['total_milestones']}\n"
        output += f"- Life Events: {summary['total_life_events']}\n"
        output += f"- Concerns: {summary['total_concerns']}\n"
        output += f"- Overdue Actions: {summary['total_overdue_actions']}\n\n"
        
        output += "### 💡 Proactive Suggestions\n\n"
        if summary['total_milestones'] > 0:
            output += f"- {summary['total_milestones']} clients have milestone birthdays coming up - great opportunity for planning discussions\n"
        if summary['total_life_events'] > 0:
            output += f"- {summary['total_life_events']} clients have life events - check if they need support\n"
        if summary['total_concerns'] > 0:
            output += f"- {summary['total_concerns']} unresolved concerns - follow up to show you're listening\n"
        
        return output
    
    def _get_client_clusters(self, query: str = "") -> str:
        """Group clients by similar characteristics and identify patterns"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        clusters = {
            'approaching_retirement': [],
            'young_professionals': [],
            'business_owners': [],
            'high_net_worth': [],
            'at_risk': []
        }
        
        for client_id, client in data["clients"].items():
            age = client.get('age', 0)
            portfolio_value = client.get("portfolio_value_gbp", 0)
            is_business_owner = client.get('is_business_owner', False)
            days_since_contact = client.get('days_since_contact', 0)
            days_since_review = client.get('days_since_review', 0)
            
            # Cluster logic
            if 60 <= age <= 65 and portfolio_value > 200000:
                clusters['approaching_retirement'].append({
                    'name': client['name'],
                    'age': age,
                    'portfolio_value': portfolio_value,
                    'has_estate_planning': client.get('has_estate_planning', False)
                })
            
            if 30 <= age <= 40 and portfolio_value < 100000:
                clusters['young_professionals'].append({
                    'name': client['name'],
                    'age': age,
                    'has_protection': client.get('has_life_insurance', False),
                    'num_dependents': client.get('num_dependents', 0)
                })
            
            if is_business_owner:
                clusters['business_owners'].append({
                    'name': client['name'],
                    'business_type': client.get('business_type', 'Unknown'),
                    'portfolio_value': portfolio_value
                })
            
            if portfolio_value > 1000000:
                clusters['high_net_worth'].append({
                    'name': client['name'],
                    'portfolio_value': portfolio_value,
                    'has_estate_planning': client.get('has_estate_planning', False)
                })
            
            if days_since_contact > 180 or days_since_review > 450:
                clusters['at_risk'].append({
                    'name': client['name'],
                    'days_since_contact': days_since_contact,
                    'days_since_review': days_since_review
                })
        
        # Format output
        output = "## 🎯 Client Clusters & Pattern Recognition\n\n"
        
        if clusters['approaching_retirement']:
            output += f"### 👴 Approaching Retirement (60-65, High Portfolio Value)\n"
            output += f"**{len(clusters['approaching_retirement'])} clients**\n\n"
            no_estate = [c for c in clusters['approaching_retirement'] if not c['has_estate_planning']]
            if no_estate:
                output += f"⚠️ **{len(no_estate)} clients without estate planning** - High-value opportunity\n"
                for client in no_estate[:5]:
                    output += f"- {client['name']} (£{client['portfolio_value']:,.0f})\n"
            output += "\n"
        
        if clusters['young_professionals']:
            output += f"### 👔 Young Professionals (30-40, Building Wealth)\n"
            output += f"**{len(clusters['young_professionals'])} clients**\n\n"
            no_protection = [c for c in clusters['young_professionals'] if c['num_dependents'] > 0 and not c['has_protection']]
            if no_protection:
                output += f"⚠️ **{len(no_protection)} clients with dependents but no protection**\n"
                for client in no_protection[:5]:
                    output += f"- {client['name']} ({client['num_dependents']} dependents)\n"
            output += "\n"
        
        if clusters['business_owners']:
            output += f"### 🏢 Business Owners\n"
            output += f"**{len(clusters['business_owners'])} clients**\n\n"
            for client in clusters['business_owners'][:5]:
                output += f"- {client['name']} ({client['business_type']})\n"
            output += "\n"
        
        if clusters['high_net_worth']:
            output += f"### 💎 High Net Worth (>£1M)\n"
            output += f"**{len(clusters['high_net_worth'])} clients**\n\n"
            no_estate = [c for c in clusters['high_net_worth'] if not c['has_estate_planning']]
            if no_estate:
                output += f"⚠️ **{len(no_estate)} clients without estate planning**\n"
            output += "\n"
        
        if clusters['at_risk']:
            output += f"### 🔴 At-Risk Clients (Contact Gaps or Overdue Reviews)\n"
            output += f"**{len(clusters['at_risk'])} clients need immediate attention**\n\n"
            for client in clusters['at_risk'][:10]:
                output += f"- {client['name']} (Contact: {client['days_since_contact']}d, Review: {client['days_since_review']}d)\n"
            output += "\n"
        
        return output
    
    def _get_value_demonstration_summary(self, query: str = "") -> str:
        """Track and quantify value delivered to clients"""
        with open("mock_data.json", "r") as f:
            data = json.load(f)
        
        total_isa_allowance_available = 0
        total_annual_allowance_available = 0
        protection_gaps_filled = 0
        clients_with_estate_planning = 0
        total_portfolio_value = 0
        
        for client_id, client in data["clients"].items():
            total_isa_allowance_available += client.get('isa_allowance_available', 0)
            total_annual_allowance_available += client.get('annual_allowance_available', 0)
            total_portfolio_value += client.get('portfolio_value_gbp', 0)
            
            if client.get('has_life_insurance') and client.get('num_dependents', 0) > 0:
                protection_gaps_filled += 1
            
            if client.get('has_estate_planning'):
                clients_with_estate_planning += 1
        
        # Calculate potential tax savings
        # ISA: Assume 5% return, 20% tax on interest/dividends
        isa_tax_savings = (total_isa_allowance_available * 0.05) * 0.20
        
        # Pension: Average 25% tax relief
        pension_tax_relief = total_annual_allowance_available * 0.25
        
        total_value = isa_tax_savings + pension_tax_relief
        
        # Get compliance stats
        reviews_due = self.compliance_tracker.get_reviews_due()
        compliance_rate = ((200 - len(reviews_due)) / 200 * 100) if len(reviews_due) < 200 else 0
        
        output = "## 📊 Value Demonstration Summary\n\n"
        output += "### 💰 Tax Savings & Optimizations\n\n"
        output += f"- **ISA Allowance Available**: £{total_isa_allowance_available:,.0f}\n"
        output += f"- **Potential Annual Tax Savings**: £{isa_tax_savings:,.0f}/year\n"
        output += f"- **Pension Annual Allowance Available**: £{total_annual_allowance_available:,.0f}\n"
        output += f"- **Potential Tax Relief**: £{pension_tax_relief:,.0f}\n"
        output += f"- **Total Potential Annual Value**: £{total_value:,.0f}\n\n"
        
        output += "### 🛡️ Risk Mitigation\n\n"
        output += f"- **Clients with Protection Coverage**: {protection_gaps_filled}\n"
        output += f"- **Clients with Estate Planning**: {clients_with_estate_planning}\n"
        output += f"- **Total Portfolio Value Managed**: £{total_portfolio_value:,.0f}\n\n"
        
        output += "### ✅ Compliance & Relationship Management\n\n"
        output += f"- **Reviews Completed on Time**: {200 - len(reviews_due)}/200\n"
        output += f"- **Compliance Rate**: {compliance_rate:.1f}%\n"
        output += f"- **Active Client Relationships**: 200\n\n"
        
        output += "### 📈 Key Metrics\n\n"
        output += f"- Average Portfolio Value: £{total_portfolio_value/200:,.0f}\n"
        output += f"- Average ISA Allowance Available: £{total_isa_allowance_available/200:,.0f}\n"
        output += f"- Average Pension Allowance Available: £{total_annual_allowance_available/200:,.0f}\n"
        
        return output


if __name__ == "__main__":
    # Test the agent
    agent = JarvisAgent()
    
    print("Jarvis: Hello! I'm Jarvis, your proactive assistant. How can I help you today?")
    print("(Type 'quit' to exit)\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Jarvis: Goodbye! Have a productive day!")
            break
        
        response = agent.chat(user_input)
        print(f"\nJarvis: {response}\n")

