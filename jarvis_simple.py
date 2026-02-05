"""Simplified Jarvis agent without LangGraph (fallback version)"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from typing import List

from config import OPENAI_API_KEY, OPENAI_MODEL
from vector_store import ClientVectorStore
from compliance_tracker import ComplianceTracker


class JarvisAgentSimple:
    """Simplified proactive chatbot agent (without LangGraph)"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
        self.vector_store = ClientVectorStore()
        self.compliance_tracker = ComplianceTracker()
        
        # Define tools
        self.tools = [
            Tool(
                name="search_clients",
                func=self._search_clients,
                description="Search for client information across all clients."
            ),
            Tool(
                name="get_client_details",
                func=self._get_client_details,
                description="Get detailed information about a specific client by name."
            ),
            Tool(
                name="get_daily_briefing",
                func=self._get_daily_briefing,
                description="Get today's proactive briefing with reviews due, contact gaps, milestones, life events."
            ),
            Tool(
                name="get_reviews_due",
                func=self._get_reviews_due,
                description="Get list of clients due for annual review."
            ),
            Tool(
                name="get_upcoming_milestones",
                func=self._get_upcoming_milestones,
                description="Get clients with upcoming milestone birthdays."
            ),
            Tool(
                name="get_life_events",
                func=self._get_life_events,
                description="Get clients with recent or upcoming life events."
            ),
            Tool(
                name="get_unresolved_concerns",
                func=self._get_unresolved_concerns,
                description="Get clients with active concerns that haven't been addressed."
            )
        ]
    
    def _determine_tool(self, query: str) -> str:
        """Determine which tool to use based on query"""
        query_lower = query.lower()
        
        if "briefing" in query_lower or "what needs" in query_lower or "attention" in query_lower:
            return "get_daily_briefing"
        elif "review" in query_lower and "due" in query_lower:
            return "get_reviews_due"
        elif "milestone" in query_lower or "birthday" in query_lower:
            return "get_upcoming_milestones"
        elif "life event" in query_lower:
            return "get_life_events"
        elif "concern" in query_lower:
            return "get_unresolved_concerns"
        elif any(word in query_lower for word in ["tell me about", "who is", "client"]):
            # Check if specific client name mentioned
            for tool in self.tools:
                if tool.name == "get_client_details":
                    # Try to extract client name (simplified)
                    return "search_clients"
        else:
            return "search_clients"
    
    def chat(self, message: str) -> str:
        """Main chat interface (simplified without LangGraph)"""
        # System prompt
        system_prompt = SystemMessage(content="""
You are Jarvis, a proactive AI assistant for UK Financial Advisors. Your role is to help advisors be proactive rather than reactive.

Key responsibilities:
1. Provide daily briefings on what needs attention (reviews due, contact gaps, milestones, life events)
2. Answer questions about clients using your knowledge base
3. Surface important information that advisors might have forgotten
4. Help with compliance tracking (Consumer Duty, annual reviews)
5. Identify opportunities (milestone birthdays, life events, tax planning windows)

Your personality:
- Professional but friendly
- Proactive and helpful
- Focused on actionable insights
- Aware of regulatory requirements (FCA, Consumer Duty)
- Understands the advisor's context (200 clients, time constraints)

When answering:
- Be concise but comprehensive
- Prioritize urgent/important items
- Provide context and reasoning
- Suggest specific actions when appropriate
- Reference specific clients and dates when relevant

Always provide actionable insights.
""")
        
        # Determine which tool to use
        tool_name = self._determine_tool(message)
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        # Execute tool if found
        tool_result = ""
        if tool:
            try:
                tool_result = tool.func(message if tool_name != "get_daily_briefing" else "")
            except Exception as e:
                tool_result = f"Error executing tool: {e}"
        
        # Prepare messages for LLM
        messages = [
            system_prompt,
            HumanMessage(content=f"User question: {message}\n\nTool result: {tool_result}\n\nPlease provide a helpful response based on the tool result."),
        ]
        
        # Get response from LLM
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            # Fallback: return tool result directly
            return tool_result if tool_result else f"I encountered an error: {e}"
    
    # Tool functions (same as in jarvis_graph.py)
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
            for meeting in meetings[:3]:
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
        
        if briefing["reviews_due"]:
            response += "TOP REVIEWS DUE:\n"
            for review in briefing["reviews_due"][:5]:
                response += f"  • {review['client_name']} - {review['days_overdue']} days overdue\n"
            response += "\n"
        
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

