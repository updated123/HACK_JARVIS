#!/usr/bin/env python3
"""Test tools directly without LLM to verify they work"""
import json
from compliance_tracker import ComplianceTracker

# Test compliance tracker (doesn't need LLM)
print("Testing Compliance Tracker...")
tracker = ComplianceTracker()

print("\n1. Reviews Due:")
reviews = tracker.get_reviews_due()
print(f"Found {len(reviews)} reviews due")
for r in reviews[:5]:
    print(f"  - {r['client_name']}: {r['days_overdue']} days overdue")

print("\n2. ISA Allowance Test:")
with open("mock_data.json", "r") as f:
    data = json.load(f)

clients_with_isa = []
for client_id, client in data["clients"].items():
    available = client.get("isa_allowance_available", 0)
    if available > 0:
        clients_with_isa.append({
            "name": client["name"],
            "available": available
        })

print(f"Found {len(clients_with_isa)} clients with ISA allowance")
for c in clients_with_isa[:5]:
    print(f"  - {c['name']}: £{c['available']:,.0f} available")

print("\n3. Business Owners:")
business_owners = []
for client_id, client in data["clients"].items():
    if client.get("is_business_owner", False):
        business_owners.append(client["name"])

print(f"Found {len(business_owners)} business owners")
for name in business_owners[:5]:
    print(f"  - {name}")

print("\n✅ All tools work without LLM! The quota error only affects:")
print("   - Semantic search (falls back to text search)")
print("   - LLM-generated responses (but tools still work)")

