"""Vector store for client information retrieval using improved text search"""
import os
import json
import pickle
from typing import List, Dict
from pathlib import Path
import re
from collections import Counter
from difflib import SequenceMatcher

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a dummy Document class for type hints when langchain is not available
    from typing import Any
    class Document:
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from config import VECTOR_STORE_PATH


class ClientVectorStore:
    """Manages vector storage and retrieval of client information using improved text search"""
    
    def __init__(self):
        self.text_splitter = None
        self.documents = []
        self.metadata_list = []
        self.index_path = Path(VECTOR_STORE_PATH) / "documents_data.pkl"
        
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
        
        # Load existing documents if available
        self._load_documents()
    
    def _load_documents(self):
        """Load existing documents if available"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'rb') as f:
                    self.documents, self.metadata_list = pickle.load(f)
                print(f"Loaded {len(self.documents)} documents from cache")
            except Exception as e:
                print(f"Warning: Could not load cached documents: {e}")
    
    def load_client_data(self, data_path: str = "mock_data.json"):
        """Load and index client data from JSON file"""
        with open(data_path, "r") as f:
            data = json.load(f)
        
        documents = []
        
        # Create documents for each client
        for client_id, client in data["clients"].items():
            client_text = self._format_client_profile(client)
            documents.append(Document(
                page_content=client_text,
                metadata={
                    "type": "client_profile",
                    "client_id": client_id,
                    "client_name": client["name"],
                    "source": "client_database"
                }
            ))
        
        # Meeting notes documents
        for meeting in data.get("meetings", []):
            meeting_text = self._format_meeting(meeting)
            documents.append(Document(
                page_content=meeting_text,
                metadata={
                    "type": "meeting_note",
                    "client_id": meeting.get("client_id"),
                    "client_name": meeting.get("client_name"),
                    "meeting_id": meeting.get("meeting_id"),
                    "date": meeting.get("date"),
                    "source": "meeting_transcript"
                }
            ))
        
        # Split documents into chunks
        all_chunks = []
        if self.text_splitter:
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata.copy()
                    ))
        else:
            all_chunks = documents
        
        # Store documents
        self.documents = all_chunks
        self.metadata_list = [doc.metadata for doc in all_chunks]
        
        # Save to disk
        Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump((self.documents, self.metadata_list), f)
        
        print(f"Indexed {len(documents)} documents ({len(all_chunks)} chunks) for {len(data['clients'])} clients")
    
    def _format_client_profile(self, client: Dict) -> str:
        """Format client profile as searchable text"""
        text = f"""
Client Profile: {client['name']}
Client ID: {client['client_id']}
Age: {client['age']} (Date of Birth: {client['date_of_birth']})
Next Birthday: {client['next_birthday']} ({client['days_to_birthday']} days away)
Milestone Birthday: {'Yes' if client.get('is_milestone_birthday') else 'No'}

Contact Information:
- Email: {client['email']}
- Phone: {client['phone']}
- Address: {client['address']}

Personal Details:
- Occupation: {client['occupation']}
- Marital Status: {client['marital_status']}
- Risk Profile: {client['risk_profile']}
- Business Owner: {'Yes' if client.get('is_business_owner') else 'No'}
- Business Type: {client.get('business_type', 'N/A')}
- High Net Worth: {'Yes' if client.get('is_high_net_worth') else 'No'}
- Retired: {'Yes' if client.get('is_retired') else 'No'}
- Retirement Age: {client.get('retirement_age', 'N/A')}

Family:
- Number of Children: {client.get('num_children', 0)}
- Children Ages: {client.get('children_ages', [])}
- Dependents: {client.get('num_dependents', 0)}
- Education Planning: {'Yes' if client.get('has_education_planning') else 'No'}

Financial Products:
{', '.join(client['products'])}

Portfolio Value: £{client['portfolio_value_gbp']:,.2f}
Equity Allocation: {client.get('equity_allocation_percent', 0):.1f}% (Target: {client.get('target_equity_allocation_min', 0)}-{client.get('target_equity_allocation_max', 0)}%)
Time Horizon: {client.get('time_horizon_years', 0)} years

Allowances:
- ISA Allowance Used: £{client.get('isa_allowance_used', 0):,.0f}
- ISA Allowance Available: £{client.get('isa_allowance_available', 0):,.0f}
- Annual Allowance Limit: £{client.get('annual_allowance_limit', 60000):,.0f}
- Annual Allowance Used: £{client.get('annual_allowance_used', 0):,.0f}
- Annual Allowance Available: £{client.get('annual_allowance_available', 0):,.0f}

Cash & Expenditure:
- Cash Holdings: £{client.get('cash_holdings_gbp', 0):,.0f}
- Monthly Expenditure: £{client.get('monthly_expenditure_gbp', 0):,.0f}
- Cash Excess: {client.get('cash_excess_months', 0):.1f} months

Retirement Planning:
- Retirement Income Goal: £{client.get('retirement_income_goal_monthly', 0):,.0f}/month
- Current Retirement Income: £{client.get('current_retirement_income_monthly', 0):,.0f}/month
- Withdrawal Rate: {client.get('withdrawal_rate_percent', 0):.2f}%

Protection:
- Life Insurance: {'Yes' if client.get('has_life_insurance') else 'No'}
- Critical Illness: {'Yes' if client.get('has_critical_illness') else 'No'}
- Income Protection: {'Yes' if client.get('has_income_protection') else 'No'}
- Protection Coverage: £{client.get('protection_coverage_amount', 0):,.0f}
- Protection Gap: {'Yes' if client.get('has_protection_gap') else 'No'}

Estate Planning:
- Estate Planning: {'Yes' if client.get('has_estate_planning') else 'No'}
- Will: {'Yes' if client.get('has_will') else 'No'}
- Trust: {'Yes' if client.get('has_trust') else 'No'}

Last Contact: {client['last_contact_date']} ({client['days_since_contact']} days ago)
Last Annual Review: {client['last_annual_review']} ({client['days_since_review']} days ago)
Review Due: {'Yes' if client['review_due'] else 'No'}

Life Events:
"""
        for event in client.get('life_events', []):
            text += f"- {event['type']} (Date: {event['date']}, Mentioned: {event['mentioned_in']})\n"
        
        text += "\nClient Concerns:\n"
        for concern in client.get('concerns', []):
            text += f"- {concern['concern']} (Status: {concern['status']}, First mentioned: {concern['first_mentioned']})\n"
        
        return text
    
    def _format_meeting(self, meeting: Dict) -> str:
        """Format meeting note as searchable text"""
        text = f"""
Meeting: {meeting['type']}
Date: {meeting['date']}
Client: {meeting['client_name']}
Duration: {meeting['duration_minutes']} minutes

Transcript:
{meeting['transcript']}

Topics Discussed:
"""
        for topic in meeting.get('topics_discussed', []):
            text += f"- {topic}\n"
        
        text += "\nRecommendations Made:\n"
        for rec in meeting.get('recommendations', []):
            text += f"- {rec.get('recommendation', '')} (Type: {rec.get('type', '')}, Status: {rec.get('status', '')})\n"
            text += f"  Rationale: {rec.get('rationale', '')}\n"
        
        text += "\nPlatforms/Products Mentioned:\n"
        for platform in meeting.get('platforms_mentioned', []):
            text += f"- {platform}\n"
        
        text += "\nConcerns Raised:\n"
        for concern in meeting.get('concerns_raised', []):
            text += f"- {concern}\n"
        
        text += "\nAction Items:\n"
        for action in meeting.get('action_items', []):
            text += f"- {action['action']} (Due: {action['due_date']}, Status: {action['status']})\n"
        
        text += "\nFollow-ups:\n"
        for follow_up in meeting.get('follow_ups', []):
            text += f"- {follow_up['type']} (Scheduled: {follow_up['scheduled_date']}, Status: {follow_up['status']})\n"
        
        text += "\nPromises Made:\n"
        for promise in meeting.get('promises', []):
            text += f"- {promise.get('promise', '')} (Promised: {promise.get('promised_date', '')}, Status: {promise.get('status', '')})\n"
        
        text += "\nDocuments Waiting For:\n"
        for doc in meeting.get('documents_waiting', []):
            text += f"- {doc.get('document_type', '')} (Requested: {doc.get('requested_date', '')})\n"
        
        return text
    
    def _calculate_relevance_score(self, query: str, document: Document) -> float:
        """Calculate relevance score using multiple methods (not just keyword matching)"""
        query_lower = query.lower()
        content_lower = document.page_content.lower()
        
        # Method 1: Exact phrase matches (highest weight)
        phrase_score = 0
        query_words = query_lower.split()
        if len(query_words) > 1:
            # Check for multi-word phrases
            for i in range(len(query_words) - 1):
                phrase = " ".join(query_words[i:i+2])
                if phrase in content_lower:
                    phrase_score += 10
        
        # Method 2: Individual word matches with frequency
        word_matches = 0
        query_word_counts = Counter(query_words)
        content_word_counts = Counter(content_lower.split())
        
        for word, count in query_word_counts.items():
            if len(word) > 2:  # Ignore very short words
                if word in content_word_counts:
                    # Weight by frequency and word importance
                    word_matches += content_word_counts[word] * (2 if len(word) > 4 else 1)
        
        # Method 3: Semantic similarity using string similarity
        similarity_score = SequenceMatcher(None, query_lower, content_lower[:200]).ratio() * 5
        
        # Method 4: Field-specific matching (boost for matches in important fields)
        field_boost = 0
        important_fields = ['client_name', 'concern', 'event_type', 'recommendation', 'topic']
        for field in important_fields:
            if field in document.metadata:
                field_value = str(document.metadata[field]).lower()
                if any(word in field_value for word in query_words if len(word) > 2):
                    field_boost += 5
        
        # Method 5: Proximity scoring (words appearing close together score higher)
        proximity_score = 0
        if len(query_words) > 1:
            positions = []
            for word in query_words:
                if len(word) > 2:
                    pos = content_lower.find(word)
                    if pos != -1:
                        positions.append(pos)
            if len(positions) > 1:
                # Calculate average distance between words
                positions.sort()
                avg_distance = sum(positions[i+1] - positions[i] for i in range(len(positions)-1)) / (len(positions)-1)
                if avg_distance < 100:  # Words close together
                    proximity_score = 10 - (avg_distance / 10)
        
        # Combine scores
        total_score = phrase_score + word_matches + similarity_score + field_boost + proximity_score
        
        return total_score
    
    def search(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[Document]:
        """Search for relevant client information using improved scoring"""
        if not self.documents:
            return []
        
        # Calculate relevance scores for all documents
        scored_docs = []
        for doc in self.documents:
            # Apply filter if provided
            if filter_dict:
                match = all(doc.metadata.get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue
            
            score = self._calculate_relevance_score(query, doc)
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]
    
    def search_by_client(self, client_name: str, query: str = None, k: int = 10) -> List[Document]:
        """Search for information about a specific client"""
        if query:
            search_query = f"{client_name} {query}"
        else:
            search_query = client_name
        
        return self.search(search_query, k=k, filter_dict={"client_name": client_name})


if __name__ == "__main__":
    # Test vector store
    store = ClientVectorStore()
    store.load_client_data("mock_data.json")
    
    # Test search
    results = store.search("client worried about inheritance tax")
    print(f"\nFound {len(results)} relevant documents")
    for i, doc in enumerate(results[:3], 1):
        print(f"\n{i}. {doc.metadata.get('client_name')} - {doc.metadata.get('type')}")
        print(doc.page_content[:200] + "...")
