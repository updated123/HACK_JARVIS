"""Compliance tracking and alerting system"""
from datetime import datetime, timedelta
from typing import List, Dict
import json
from config import (
    ANNUAL_REVIEW_PERIOD_DAYS,
    CONTACT_GAP_WARNING_DAYS,
    MILESTONE_BIRTHDAY_AGES,
    DAYS_AHEAD_FOR_ALERTS
)


class ComplianceTracker:
    """Tracks compliance requirements and generates proactive alerts"""
    
    def __init__(self, data_path: str = "mock_data.json"):
        with open(data_path, "r") as f:
            self.data = json.load(f)
    
    def get_reviews_due(self) -> List[Dict]:
        """Get clients due for annual review"""
        reviews_due = []
        
        for client_id, client in self.data["clients"].items():
            days_since = client["days_since_review"]
            
            if days_since >= ANNUAL_REVIEW_PERIOD_DAYS:
                reviews_due.append({
                    "client_id": client_id,
                    "client_name": client["name"],
                    "days_overdue": days_since - ANNUAL_REVIEW_PERIOD_DAYS,
                    "last_review": client["last_annual_review"],
                    "priority": "high" if days_since > ANNUAL_REVIEW_PERIOD_DAYS + 30 else "medium"
                })
        
        return sorted(reviews_due, key=lambda x: x["days_overdue"], reverse=True)
    
    def get_contact_gaps(self) -> List[Dict]:
        """Get clients who haven't been contacted recently"""
        contact_gaps = []
        
        for client_id, client in self.data["clients"].items():
            days_since = client["days_since_contact"]
            
            if days_since >= CONTACT_GAP_WARNING_DAYS:
                contact_gaps.append({
                    "client_id": client_id,
                    "client_name": client["name"],
                    "days_since_contact": days_since,
                    "last_contact": client["last_contact_date"],
                    "priority": "high" if days_since > 180 else "medium"
                })
        
        return sorted(contact_gaps, key=lambda x: x["days_since_contact"], reverse=True)
    
    def get_upcoming_milestones(self) -> List[Dict]:
        """Get clients with upcoming milestone birthdays"""
        milestones = []
        today = datetime.now()
        
        for client_id, client in self.data["clients"].items():
            if client.get("is_milestone_birthday") and client["days_to_birthday"] <= DAYS_AHEAD_FOR_ALERTS:
                next_age = client["age"] + (1 if client["days_to_birthday"] < 365 else 0)
                
                milestones.append({
                    "client_id": client_id,
                    "client_name": client["name"],
                    "current_age": client["age"],
                    "turning_age": next_age,
                    "days_until": client["days_to_birthday"],
                    "birthday_date": client["next_birthday"],
                    "opportunity": f"Pension access at {next_age} (if applicable), tax planning opportunities"
                })
        
        return sorted(milestones, key=lambda x: x["days_until"])
    
    def get_life_events_requiring_attention(self) -> List[Dict]:
        """Get clients with recent or upcoming life events"""
        life_events = []
        today = datetime.now()
        
        for client_id, client in self.data["clients"].items():
            for event in client.get("life_events", []):
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                days_until = (event_date - today).days
                
                # Include events in the past 90 days or next 90 days
                if -90 <= days_until <= 90:
                    life_events.append({
                        "client_id": client_id,
                        "client_name": client["name"],
                        "event_type": event["type"],
                        "event_date": event["date"],
                        "days_until": days_until,
                        "mentioned_in": event.get("mentioned_in", "Unknown"),
                        "action_required": "Review financial plan, update beneficiaries, tax planning"
                    })
        
        return sorted(life_events, key=lambda x: abs(x["days_until"]))
    
    def get_unresolved_concerns(self) -> List[Dict]:
        """Get clients with active concerns that haven't been addressed"""
        concerns = []
        
        for client_id, client in self.data["clients"].items():
            for concern in client.get("concerns", []):
                if concern["status"] in ["active", "monitoring"]:
                    concerns.append({
                        "client_id": client_id,
                        "client_name": client["name"],
                        "concern": concern["concern"],
                        "status": concern["status"],
                        "first_mentioned": concern["first_mentioned"],
                        "action_required": "Follow up to address concern, provide reassurance or action plan"
                    })
        
        return concerns
    
    def get_overdue_action_items(self) -> List[Dict]:
        """Get overdue action items from meetings"""
        overdue_actions = []
        today = datetime.now()
        
        for meeting in self.data["meetings"]:
            for action in meeting.get("action_items", []):
                if action["status"] != "completed":
                    due_date = datetime.strptime(action["due_date"], "%Y-%m-%d")
                    if due_date < today:
                        overdue_actions.append({
                            "client_id": meeting["client_id"],
                            "client_name": meeting["client_name"],
                            "action": action["action"],
                            "due_date": action["due_date"],
                            "days_overdue": (today - due_date).days,
                            "meeting_date": meeting["date"],
                            "assigned_to": action.get("assigned_to", "Advisor")
                        })
        
        return sorted(overdue_actions, key=lambda x: x["days_overdue"], reverse=True)
    
    def get_overdue_follow_ups(self) -> List[Dict]:
        """Get overdue follow-up commitments"""
        overdue_follow_ups = []
        today = datetime.now()
        
        for meeting in self.data["meetings"]:
            for follow_up in meeting.get("follow_ups", []):
                if follow_up["status"] == "overdue":
                    scheduled_date = datetime.strptime(follow_up["scheduled_date"], "%Y-%m-%d")
                    overdue_follow_ups.append({
                        "client_id": meeting["client_id"],
                        "client_name": meeting["client_name"],
                        "follow_up_type": follow_up["type"],
                        "scheduled_date": follow_up["scheduled_date"],
                        "days_overdue": (today - scheduled_date).days,
                        "original_meeting_date": meeting["date"]
                    })
        
        return sorted(overdue_follow_ups, key=lambda x: x["days_overdue"], reverse=True)
    
    def get_daily_briefing(self) -> Dict:
        """Generate comprehensive daily briefing"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "reviews_due": self.get_reviews_due(),
            "contact_gaps": self.get_contact_gaps(),
            "upcoming_milestones": self.get_upcoming_milestones(),
            "life_events": self.get_life_events_requiring_attention(),
            "unresolved_concerns": self.get_unresolved_concerns(),
            "overdue_actions": self.get_overdue_action_items(),
            "overdue_follow_ups": self.get_overdue_follow_ups(),
            "summary": {
                "total_reviews_due": len(self.get_reviews_due()),
                "total_contact_gaps": len(self.get_contact_gaps()),
                "total_milestones": len(self.get_upcoming_milestones()),
                "total_life_events": len(self.get_life_events_requiring_attention()),
                "total_concerns": len(self.get_unresolved_concerns()),
                "total_overdue_actions": len(self.get_overdue_action_items()),
                "total_overdue_follow_ups": len(self.get_overdue_follow_ups())
            }
        }


if __name__ == "__main__":
    tracker = ComplianceTracker()
    briefing = tracker.get_daily_briefing()
    
    print("Daily Compliance Briefing")
    print("=" * 50)
    print(f"\nReviews Due: {briefing['summary']['total_reviews_due']}")
    print(f"Contact Gaps: {briefing['summary']['total_contact_gaps']}")
    print(f"Upcoming Milestones: {briefing['summary']['total_milestones']}")
    print(f"Life Events: {briefing['summary']['total_life_events']}")
    print(f"Unresolved Concerns: {briefing['summary']['total_concerns']}")
    print(f"Overdue Actions: {briefing['summary']['total_overdue_actions']}")
    print(f"Overdue Follow-ups: {briefing['summary']['total_overdue_follow_ups']}")

