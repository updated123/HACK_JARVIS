import { useState } from 'react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([
    {
      role: 'assistant',
      content: "Hello! I'm Jarvis, your proactive assistant. I help you stay on top of client relationships, compliance, and opportunities.\n\nTry asking me:\n• 'What needs my attention today?'\n• 'Show me reviews due'\n• 'Who has upcoming milestone birthdays?'"
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [briefing, setBriefing] = useState<any>(null);

    const sendMessage = async () => {
        if (!message.trim()) return;
        
        const userMessage = { role: 'user', content: message };
        setMessages(prev => [...prev, userMessage]);
        const currentMessage = message;
        setMessage('');
        setLoading(true);

        try {
            // Send conversation history for context
            const conversationHistory = messages.map(msg => ({
                role: msg.role,
                content: msg.content
            }));
            
            const response = await axios.post(`${API_URL}/api/chat`, {
                message: currentMessage,
                conversation_history: conversationHistory
            });
            
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.data.response
            }]);
        } catch (error: any) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${error.response?.data?.detail || error.message}`
            }]);
        } finally {
            setLoading(false);
        }
    };

  const loadBriefing = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/briefing`);
      setBriefing(response.data);
    } catch (error: any) {
      alert(`Error loading briefing: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <h1>🤖 AdvisoryAI Jarvis</h1>
      <p>Proactive Assistant for Financial Advisors</p>

      <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
        <div style={{ flex: 1 }}>
          <h2>💬 Chat with Jarvis</h2>
          
          <div style={{ 
            border: '1px solid #ccc', 
            borderRadius: '8px', 
            padding: '20px', 
            minHeight: '400px',
            marginBottom: '20px',
            overflowY: 'auto'
          }}>
            {messages.map((msg, idx) => (
              <div key={idx} style={{ 
                marginBottom: '15px',
                padding: '10px',
                backgroundColor: msg.role === 'user' ? '#e3f2fd' : '#f5f5f5',
                borderRadius: '8px'
              }}>
                <strong>{msg.role === 'user' ? 'You' : 'Jarvis'}:</strong>
                <div style={{ marginTop: '5px', whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </div>
              </div>
            ))}
            {loading && <div>Thinking...</div>}
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask Jarvis anything..."
              style={{ flex: 1, padding: '10px', fontSize: '16px' }}
            />
            <button 
              onClick={sendMessage}
              disabled={loading}
              style={{ padding: '10px 20px', fontSize: '16px' }}
            >
              Send
            </button>
          </div>

          <div style={{ marginTop: '20px' }}>
            <button onClick={loadBriefing} disabled={loading}>
              📊 Load Daily Briefing
            </button>
          </div>
        </div>

        {briefing && (
          <div style={{ flex: 1 }}>
            <h2>📋 Daily Briefing</h2>
            <div style={{ border: '1px solid #ccc', borderRadius: '8px', padding: '20px' }}>
              <h3>Summary</h3>
              <p>Reviews Due: {briefing.summary?.total_reviews_due || 0}</p>
              <p>Contact Gaps: {briefing.summary?.total_contact_gaps || 0}</p>
              <p>Overdue Actions: {briefing.summary?.total_overdue_actions || 0}</p>
              
              {briefing.reviews_due && briefing.reviews_due.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4>Reviews Due</h4>
                  {briefing.reviews_due.slice(0, 5).map((review: any, idx: number) => (
                    <div key={idx} style={{ marginBottom: '10px', padding: '10px', backgroundColor: '#fff3cd' }}>
                      <strong>{review.client_name}</strong> - {review.days_overdue} days overdue
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

