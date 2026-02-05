"""Quick demo script to test Jarvis functionality"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from compliance_tracker import ComplianceTracker
from data_generator import generate_all_clients
import json


def demo_compliance_tracker():
    """Demo the compliance tracking functionality"""
    print("=" * 60)
    print("Compliance Tracker Demo")
    print("=" * 60)
    print()
    
    # Generate data if needed
    data_file = Path("mock_data.json")
    if not data_file.exists():
        print("Generating mock data...")
        data = generate_all_clients(50)  # Smaller set for demo
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {data['total_clients']} clients\n")
    
    # Initialize tracker
    tracker = ComplianceTracker()
    
    # Get briefing
    briefing = tracker.get_daily_briefing()
    summary = briefing["summary"]
    
    print("Daily Briefing Summary:")
    print(f"  Reviews Due: {summary['total_reviews_due']}")
    print(f"  Contact Gaps: {summary['total_contact_gaps']}")
    print(f"  Overdue Actions: {summary['total_overdue_actions']}")
    print(f"  Upcoming Milestones: {summary['total_milestones']}")
    print(f"  Life Events: {summary['total_life_events']}")
    print(f"  Unresolved Concerns: {summary['total_concerns']}")
    print()
    
    # Show top reviews due
    if briefing["reviews_due"]:
        print("Top 5 Reviews Due:")
        for review in briefing["reviews_due"][:5]:
            print(f"  • {review['client_name']} - {review['days_overdue']} days overdue")
        print()
    
    # Show milestones
    if briefing["upcoming_milestones"]:
        print("Upcoming Milestones:")
        for milestone in briefing["upcoming_milestones"][:5]:
            print(f"  • {milestone['client_name']} turning {milestone['turning_age']} in {milestone['days_until']} days")
        print()
    
    # Show life events
    if briefing["life_events"]:
        print("Life Events Requiring Attention:")
        for event in briefing["life_events"][:5]:
            days_text = f"{abs(event['days_until'])} days ago" if event['days_until'] < 0 else f"in {event['days_until']} days"
            print(f"  • {event['client_name']}: {event['event_type']} ({days_text})")
        print()


if __name__ == "__main__":
    demo_compliance_tracker()

