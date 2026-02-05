#!/usr/bin/env python3
"""Direct query interface that works without LLM"""
import sys
from jarvis_graph import JarvisAgent

# Simple keyword-based router
QUERY_ROUTES = {
    "isa": "_get_clients_with_isa_allowance",
    "annual allowance": "_get_clients_with_annual_allowance",
    "cash excess": "_get_clients_with_cash_excess",
    "underweight": "_get_clients_underweight_equities",
    "equity": "_get_clients_underweight_equities",
    "retirement": "_get_clients_retirement_trajectory_issues",
    "protection": "_get_clients_with_protection_gaps",
    "withdrawal": "_get_retired_clients_high_withdrawal",
    "review": "_get_reviews_due",
    "12 months": "_get_clients_reviews_overdue_12_months",
    "business": "_get_business_owners",
    "university": "_get_clients_university_planning",
    "estate": "_get_hnw_clients_no_estate_planning",
    "birthday": "_get_clients_birthdays_this_month",
    "documents": "_get_documents_waiting",
    "waiting": "_get_waiting_on_clients",
    "actions": "_get_open_action_items",
    "follow": "_get_overdue_follow_ups",
    "briefing": "_get_daily_briefing",
    "milestone": "_get_upcoming_milestones",
    "life event": "_get_life_events",
    "concern": "_get_unresolved_concerns"
}

def route_query(query: str):
    """Route query to appropriate tool"""
    query_lower = query.lower()
    
    # Find matching route
    for keyword, tool_name in QUERY_ROUTES.items():
        if keyword in query_lower:
            return tool_name
    
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 direct_query.py '<your query>'")
        print("\nExample queries:")
        print("  - 'Show me clients with ISA allowance'")
        print("  - 'Which clients haven't had a review in 12 months'")
        print("  - 'Show me business owners'")
        print("  - 'What documents am I waiting for'")
        print("  - 'Daily briefing'")
        return
    
    query = " ".join(sys.argv[1:])
    print(f"\nQuery: {query}\n")
    
    # Try to route query
    tool_name = route_query(query)
    
    if not tool_name:
        print("❌ Could not route query. Try one of these:")
        for keyword in QUERY_ROUTES.keys():
            print(f"  - {keyword}")
        return
    
    try:
        # Initialize agent (tools work even if LLM doesn't)
        agent = JarvisAgent()
        
        # Get the tool function
        tool_func = getattr(agent, tool_name, None)
        
        if not tool_func:
            print(f"❌ Tool {tool_name} not found")
            return
        
        # Call tool directly
        print(f"🔧 Using tool: {tool_name}\n")
        result = tool_func("")
        print(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

