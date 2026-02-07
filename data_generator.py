"""Generate realistic mock data for UK Financial Advisors"""
from faker import Faker
from datetime import datetime, timedelta
import random
import json
from typing import List, Dict
from dateutil.relativedelta import relativedelta

fake = Faker('en_GB')

# UK-specific financial products and concerns
FINANCIAL_PRODUCTS = [
    "SIPP (Self-Invested Personal Pension)",
    "ISA (Individual Savings Account)",
    "General Investment Account",
    "Annuity",
    "Life Insurance",
    "Critical Illness Cover",
    "Income Protection",
    "Bonds",
    "Unit Trusts",
    "OEICs (Open-Ended Investment Companies)"
]

LIFE_EVENTS = [
    "daughter's wedding",
    "son's graduation",
    "new grandchild",
    "upcoming retirement",
    "house purchase",
    "divorce",
    "inheritance received",
    "career change",
    "health concern",
    "holiday home purchase",
    "children moving out",
    "elderly parent care needs"
]

CLIENT_CONCERNS = [
    "anxiety about market volatility",
    "worry about inheritance tax",
    "retirement income adequacy",
    "long-term care costs",
    "pension transfer value",
    "tax efficiency",
    "estate planning",
    "inflation impact on savings",
    "NHS vs private healthcare",
    "state pension eligibility"
]

COMPLIANCE_ACTIONS = [
    "annual review",
    "risk profile update",
    "fact-find refresh",
    "suitability assessment",
    "portfolio rebalancing",
    "policy review",
    "beneficiary update",
    "address change verification"
]


def generate_client_profile(client_id: int) -> Dict:
    """Generate a realistic UK client profile"""
    age = random.randint(35, 75)
    birth_date = datetime.now() - relativedelta(years=age)
    
    # Determine if approaching milestone birthday
    next_birthday = birth_date.replace(year=datetime.now().year)
    if next_birthday < datetime.now():
        next_birthday = next_birthday.replace(year=datetime.now().year + 1)
    
    days_to_birthday = (next_birthday - datetime.now()).days
    is_milestone = age + (1 if days_to_birthday < 365 else 0) in [50, 55, 60, 65, 70, 75]
    
    # Last contact date (some clients haven't been contacted recently)
    # Generate more varied contact dates
    contact_distribution = random.random()
    if contact_distribution < 0.30:  # 30% - recent contact (within last month)
        days_since_contact = random.randint(1, 30)
    elif contact_distribution < 0.55:  # 25% - recent (1-2 months)
        days_since_contact = random.randint(30, 60)
    elif contact_distribution < 0.70:  # 15% - moderate (2-3 months)
        days_since_contact = random.randint(60, 90)
    elif contact_distribution < 0.85:  # 15% - overdue (3-6 months)
        days_since_contact = random.randint(90, 180)
    else:  # 15% - significantly overdue (6+ months)
        days_since_contact = random.randint(180, 365)
    
    last_contact = datetime.now() - timedelta(days=days_since_contact)
    
    # Last annual review - more varied dates
    # Generate a more realistic distribution with more variation
    review_distribution = random.random()
    if review_distribution < 0.15:  # 15% - recent reviews (within last 6 months)
        days_since_review = random.randint(30, 180)
    elif review_distribution < 0.35:  # 20% - approaching due (6-11 months)
        days_since_review = random.randint(180, 330)
    elif review_distribution < 0.65:  # 30% - just due or slightly overdue (11-13 months)
        days_since_review = random.randint(330, 400)
    elif review_distribution < 0.85:  # 20% - moderately overdue (13-18 months)
        days_since_review = random.randint(400, 550)
    else:  # 15% - significantly overdue (18+ months)
        days_since_review = random.randint(550, 730)
    
    last_review = datetime.now() - timedelta(days=days_since_review)
    
    # Generate products
    num_products = random.randint(1, 4)
    products = random.sample(FINANCIAL_PRODUCTS, num_products)
    
    # Risk profile
    risk_profiles = ["Conservative", "Cautious", "Balanced", "Adventurous", "Aggressive"]
    risk_profile = random.choices(
        risk_profiles,
        weights=[15, 25, 35, 20, 5]  # Most clients are balanced/cautious
    )[0]
    
    # Generate life events (some clients have recent/upcoming events)
    life_events = []
    if random.random() < 0.4:  # 40% have a life event
        event_type = random.choice(LIFE_EVENTS)
        event_date = datetime.now() + timedelta(days=random.randint(-90, 90))
        life_events.append({
            "type": event_type,
            "date": event_date.strftime("%Y-%m-%d"),
            "mentioned_in": f"Meeting on {last_contact.strftime('%Y-%m-%d')}"
        })
    
    # Generate concerns
    concerns = []
    if random.random() < 0.5:  # 50% have expressed concerns
        concern = random.choice(CLIENT_CONCERNS)
        concerns.append({
            "concern": concern,
            "first_mentioned": (last_contact - timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d"),
            "status": random.choice(["active", "addressed", "monitoring"])
        })
    
    # Investment-related fields
    portfolio_value = round(random.uniform(50000, 1500000), 2)
    
    # Equity allocation (percentage of portfolio in equities)
    # Risk profile determines target equity allocation
    risk_to_equity_target = {
        "Conservative": (20, 40),
        "Cautious": (30, 50),
        "Balanced": (50, 70),
        "Adventurous": (70, 85),
        "Aggressive": (85, 95)
    }
    target_min, target_max = risk_to_equity_target[risk_profile]
    current_equity_allocation = round(random.uniform(target_min - 20, target_max + 10), 1)  # Some deviation
    current_equity_allocation = max(0, min(100, current_equity_allocation))  # Clamp to 0-100
    
    # ISA allowance (2024/25: £20,000)
    isa_allowance_used = round(random.uniform(0, 20000), 2)
    isa_allowance_available = max(0, 20000 - isa_allowance_used)
    
    # Annual allowance (pension - 2024/25: £60,000, but can be tapered)
    annual_allowance_limit = random.choice([60000, 60000, 60000, 40000, 20000])  # Most have full, some tapered
    annual_allowance_used = round(random.uniform(0, annual_allowance_limit * 0.9), 2)
    annual_allowance_available = max(0, annual_allowance_limit - annual_allowance_used)
    
    # Cash holdings and monthly expenditure
    monthly_expenditure = round(random.uniform(2000, 8000), 2)
    cash_holdings = round(random.uniform(monthly_expenditure * 2, monthly_expenditure * 18), 2)  # 2-18 months
    cash_excess_months = cash_holdings / monthly_expenditure
    
    # Retirement planning
    is_retired = age >= 65 or (age >= 60 and random.random() < 0.3)
    retirement_age = random.randint(60, 70) if not is_retired else age
    retirement_income_goal = round(monthly_expenditure * random.uniform(0.8, 1.2), 2) if is_retired else round(monthly_expenditure * random.uniform(0.7, 1.5), 2)
    current_retirement_income = round(retirement_income_goal * random.uniform(0.6, 1.1), 2) if is_retired else 0
    withdrawal_rate = round((current_retirement_income * 12 / portfolio_value * 100), 2) if is_retired and portfolio_value > 0 else 0
    
    # Time horizon (years until retirement or life expectancy)
    time_horizon = max(1, retirement_age - age) if not is_retired else random.randint(15, 30)
    
    # Protection coverage
    has_life_insurance = random.random() < 0.6
    has_critical_illness = random.random() < 0.4
    has_income_protection = random.random() < 0.3
    protection_coverage_amount = round(random.uniform(100000, 1000000), 2) if has_life_insurance else 0
    
    # Family circumstances for protection gap analysis
    num_dependents = random.randint(0, 3) if age < 65 else random.randint(0, 1)
    has_protection_gap = num_dependents > 0 and not has_life_insurance
    
    # Business owner flag
    is_business_owner = random.random() < 0.15  # 15% are business owners
    business_type = random.choice(["Limited Company", "Partnership", "Sole Trader"]) if is_business_owner else None
    
    # Estate planning
    has_estate_planning = random.random() < 0.4 if portfolio_value > 500000 else random.random() < 0.2
    has_will = random.random() < 0.7
    has_trust = random.random() < 0.2 if portfolio_value > 500000 else False
    
    # Children information
    num_children = random.randint(0, 3)
    children_ages = [random.randint(0, 25) for _ in range(num_children)] if num_children > 0 else []
    children_approaching_university = [age for age in children_ages if 16 <= age <= 19]
    has_education_planning = len(children_approaching_university) > 0 and random.random() < 0.3
    
    # High net worth flag (typically >£1M investable assets)
    is_high_net_worth = portfolio_value > 1000000
    
    client = {
        "client_id": f"CLI{client_id:04d}",
        "name": fake.name(),
        "age": age,
        "date_of_birth": birth_date.strftime("%Y-%m-%d"),
        "next_birthday": next_birthday.strftime("%Y-%m-%d"),
        "days_to_birthday": days_to_birthday,
        "is_milestone_birthday": is_milestone and days_to_birthday < 90,
        "email": fake.email(),
        "phone": fake.phone_number(),
        "address": fake.address().replace('\n', ', '),
        "occupation": fake.job(),
        "marital_status": random.choice(["Married", "Single", "Divorced", "Widowed"]),
        "risk_profile": risk_profile,
        "products": products,
        "portfolio_value_gbp": portfolio_value,
        "last_contact_date": last_contact.strftime("%Y-%m-%d"),
        "days_since_contact": days_since_contact,
        "last_annual_review": last_review.strftime("%Y-%m-%d"),
        "days_since_review": days_since_review,
        "review_due": days_since_review >= 365,
        "life_events": life_events,
        "concerns": concerns,
        "notes": [],
        # Investment fields
        "equity_allocation_percent": current_equity_allocation,
        "target_equity_allocation_min": target_min,
        "target_equity_allocation_max": target_max,
        "time_horizon_years": time_horizon,
        "isa_allowance_used": isa_allowance_used,
        "isa_allowance_available": isa_allowance_available,
        "annual_allowance_limit": annual_allowance_limit,
        "annual_allowance_used": annual_allowance_used,
        "annual_allowance_available": annual_allowance_available,
        "cash_holdings_gbp": cash_holdings,
        "monthly_expenditure_gbp": monthly_expenditure,
        "cash_excess_months": round(cash_excess_months, 1),
        # Retirement fields
        "is_retired": is_retired,
        "retirement_age": retirement_age,
        "retirement_income_goal_monthly": retirement_income_goal,
        "current_retirement_income_monthly": current_retirement_income,
        "withdrawal_rate_percent": withdrawal_rate,
        # Protection fields
        "has_life_insurance": has_life_insurance,
        "has_critical_illness": has_critical_illness,
        "has_income_protection": has_income_protection,
        "protection_coverage_amount": protection_coverage_amount,
        "num_dependents": num_dependents,
        "has_protection_gap": has_protection_gap,
        # Business fields
        "is_business_owner": is_business_owner,
        "business_type": business_type,
        # Estate planning
        "has_estate_planning": has_estate_planning,
        "has_will": has_will,
        "has_trust": has_trust,
        "is_high_net_worth": is_high_net_worth,
        # Family fields
        "num_children": num_children,
        "children_ages": children_ages,
        "has_education_planning": has_education_planning
    }
    
    return client


def generate_meeting_notes(client: Dict, num_meetings: int = None) -> List[Dict]:
    """Generate realistic meeting notes for a client"""
    if num_meetings is None:
        num_meetings = random.randint(2, 8)
    
    meetings = []
    base_date = datetime.strptime(client["last_contact_date"], "%Y-%m-%d")
    
    for i in range(num_meetings):
        meeting_date = base_date - timedelta(days=random.randint(0, 365 * 2))
        meeting_type = random.choice([
            "Initial Consultation",
            "Annual Review",
            "Ad-hoc Discussion",
            "Product Recommendation",
            "Life Event Planning",
            "Compliance Check"
        ])
        
        # Generate realistic meeting transcript
        transcript = generate_meeting_transcript(client, meeting_type)
        
        # Generate recommendations made in this meeting
        recommendations = generate_recommendations(client, meeting_type) if random.random() < 0.7 else []
        
        # Generate platforms/products mentioned
        platforms_mentioned = random.sample(["Platform X", "Platform Y", "Platform Z", "Provider A", "Provider B"], random.randint(0, 2)) if random.random() < 0.5 else []
        
        # Generate topics discussed
        topics_discussed = []
        if "volatility" in transcript.lower() or "market" in transcript.lower():
            topics_discussed.append("market volatility")
        if "inheritance" in transcript.lower() or "tax" in transcript.lower():
            topics_discussed.append("inheritance tax")
        if "sustainable" in transcript.lower() or "ESG" in transcript.lower():
            topics_discussed.append("sustainable investing")
        if not topics_discussed:
            topics_discussed = random.sample(["pension planning", "investment strategy", "tax efficiency", "retirement planning"], random.randint(1, 2))
        
        # Generate promises/commitments made
        promises = generate_promises() if random.random() < 0.4 else []
        
        # Generate documents waiting for
        documents_waiting = generate_documents_waiting() if random.random() < 0.3 else []
        
        meeting = {
            "meeting_id": f"MTG{client['client_id']}{i+1:03d}",
            "client_id": client["client_id"],
            "client_name": client["name"],
            "date": meeting_date.strftime("%Y-%m-%d"),
            "type": meeting_type,
            "duration_minutes": random.randint(30, 90),
            "transcript": transcript,
            "action_items": generate_action_items(),
            "follow_ups": generate_follow_ups(),
            "recommendations": recommendations,
            "platforms_mentioned": platforms_mentioned,
            "topics_discussed": topics_discussed,
            "promises": promises,
            "documents_waiting": documents_waiting,
            "concerns_raised": [c["concern"] for c in client.get("concerns", []) if c["status"] == "active"]
        }
        
        meetings.append(meeting)
        client["notes"].append(meeting)
    
    return meetings


def generate_meeting_transcript(client: Dict, meeting_type: str) -> str:
    """Generate a realistic meeting transcript"""
    advisor_name = "Sarah"
    client_name = client["name"].split()[0]
    
    transcripts = {
        "Initial Consultation": f"""
{advisor_name}: Good morning {client_name}, thank you for coming in today. I understand you're looking to review your financial situation?

{client_name}: Yes, that's right. I'm approaching retirement in a few years and want to make sure everything is in order.

{advisor_name}: Absolutely. Let's start by understanding your current situation. What are your main financial goals?

{client_name}: Well, I'd like to retire comfortably, and I'm also concerned about inheritance tax for my children. My father passed away recently and left me some assets.

{advisor_name}: I'm sorry to hear about your father. That's definitely something we should address. Let's discuss your current pension arrangements and how we can structure things tax-efficiently.
""",
        "Annual Review": f"""
{advisor_name}: Hi {client_name}, great to see you again. It's been about a year since we last met. How have things been?

{client_name}: Good, thanks. Work's been busy, but I'm managing. I'm a bit worried about the markets though - they seem quite volatile.

{advisor_name}: I understand that concern. Market volatility is normal, but I can see your portfolio is well-diversified and aligned with your {client['risk_profile'].lower()} risk profile. Let's review your current position and make sure everything still aligns with your goals.

{client_name}: That would be helpful. Also, my daughter is getting married next year, so I'm thinking about how that might affect things.

{advisor_name}: Congratulations! That's wonderful news. A wedding is definitely a life event we should plan for. Let's discuss how this fits into your overall financial plan.
""",
        "Life Event Planning": f"""
{advisor_name}: {client_name}, you mentioned {random.choice(LIFE_EVENTS)}. How would you like to approach this from a financial planning perspective?

{client_name}: I'm not entirely sure. I want to make sure I'm making the right decisions and not missing any opportunities.

{advisor_name}: Of course. Let's think about the tax implications, any allowances you might be able to use, and how this fits with your long-term goals. We should also review your beneficiaries to ensure they're up to date.
"""
    }
    
    default_transcript = f"""
{advisor_name}: Hi {client_name}, thanks for coming in. What would you like to discuss today?

{client_name}: I wanted to review my {random.choice(client.get('products', ['pension']))} and see if there are any changes I should consider.

{advisor_name}: Let's take a look at your current position and see what makes sense for your situation.
"""
    
    return transcripts.get(meeting_type, default_transcript).strip()


def generate_action_items() -> List[Dict]:
    """Generate action items from a meeting"""
    actions = []
    num_actions = random.randint(0, 3)
    
    action_templates = [
        "Review pension transfer value",
        "Update beneficiary information",
        "Consider ISA allowance for this tax year",
        "Review life insurance coverage",
        "Schedule follow-up meeting",
        "Provide inheritance tax planning document",
        "Review investment portfolio allocation"
    ]
    
    for _ in range(num_actions):
        action = random.choice(action_templates)
        due_date = datetime.now() + timedelta(days=random.randint(7, 60))
        actions.append({
            "action": action,
            "assigned_to": "Advisor",
            "due_date": due_date.strftime("%Y-%m-%d"),
            "status": random.choice(["pending", "in_progress", "completed"])
        })
    
    return actions


def generate_follow_ups() -> List[Dict]:
    """Generate follow-up commitments"""
    follow_ups = []
    if random.random() < 0.6:  # 60% of meetings have follow-ups
        follow_up_date = datetime.now() + timedelta(days=random.randint(30, 180))
        follow_ups.append({
            "type": random.choice(["Check-in call", "Review meeting", "Document follow-up"]),
            "scheduled_date": follow_up_date.strftime("%Y-%m-%d"),
            "status": "scheduled" if follow_up_date > datetime.now() else "overdue"
        })
    
    return follow_ups


def generate_recommendations(client: Dict, meeting_type: str) -> List[Dict]:
    """Generate recommendations made during meetings"""
    recommendations = []
    num_recommendations = random.randint(0, 3)
    
    recommendation_templates = [
        {
            "type": "product",
            "recommendation": f"Consider increasing equity allocation to align with {client['risk_profile']} risk profile",
            "rationale": "Current allocation is below target range for risk profile and time horizon"
        },
        {
            "type": "product",
            "recommendation": "Utilize remaining ISA allowance for this tax year",
            "rationale": f"£{client.get('isa_allowance_available', 0):,.0f} ISA allowance still available"
        },
        {
            "type": "product",
            "recommendation": "Consider Platform X for better fund access and lower fees",
            "rationale": "Current platform charges are higher than market average"
        },
        {
            "type": "planning",
            "recommendation": "Review protection coverage given family circumstances",
            "rationale": f"{client.get('num_dependents', 0)} dependents identified but no life insurance in place"
        },
        {
            "type": "planning",
            "recommendation": "Consider estate planning to minimize inheritance tax",
            "rationale": "Portfolio value exceeds IHT threshold and no estate planning in place"
        },
        {
            "type": "strategy",
            "recommendation": "Rebalance portfolio to target allocation",
            "rationale": "Current allocation has drifted from target due to market movements"
        }
    ]
    
    for _ in range(num_recommendations):
        rec = random.choice(recommendation_templates)
        recommendations.append({
            "recommendation": rec["recommendation"],
            "rationale": rec["rationale"],
            "type": rec["type"],
            "status": random.choice(["accepted", "under_consideration", "declined"])
        })
    
    return recommendations


def generate_promises() -> List[Dict]:
    """Generate promises/commitments made by advisor"""
    promises = []
    if random.random() < 0.4:
        promise_date = datetime.now() + timedelta(days=random.randint(1, 30))
        promises.append({
            "promise": random.choice([
                "Send inheritance tax planning document",
                "Provide comparison of pension providers",
                "Draft cashflow projection",
                "Review protection quotes",
                "Send ISA application forms"
            ]),
            "promised_date": promise_date.strftime("%Y-%m-%d"),
            "status": "pending" if promise_date > datetime.now() else "overdue"
        })
    
    return promises


def generate_documents_waiting() -> List[Dict]:
    """Generate documents waiting for from clients"""
    documents = []
    if random.random() < 0.3:
        documents.append({
            "document_type": random.choice([
                "P60",
                "Pension statements",
                "Bank statements",
                "Property valuation",
                "Will copy",
                "Trust documents"
            ]),
            "requested_date": (datetime.now() - timedelta(days=random.randint(7, 60))).strftime("%Y-%m-%d"),
            "status": "waiting"
        })
    
    return documents


def generate_all_clients(num_clients: int = 200) -> Dict:
    """Generate all client data"""
    clients = {}
    all_meetings = []
    
    for i in range(1, num_clients + 1):
        client = generate_client_profile(i)
        meetings = generate_meeting_notes(client)
        clients[client["client_id"]] = client
        all_meetings.extend(meetings)
    
    return {
        "clients": clients,
        "meetings": all_meetings,
        "generated_at": datetime.now().isoformat(),
        "total_clients": num_clients,
        "total_meetings": len(all_meetings)
    }


if __name__ == "__main__":
    # Generate mock data
    print("Generating mock client data...")
    data = generate_all_clients(200)
    
    # Save to JSON
    with open("mock_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {data['total_clients']} clients and {data['total_meetings']} meetings")
    print("Data saved to mock_data.json")

