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

  // Helper function to format message content (basic markdown-like formatting)
  const formatMessage = (content: string) => {
    // Split by lines and format numbered lists, bold text, etc.
    const lines = content.split('\n');
    return lines.map((line, idx) => {
      // Format numbered lists (1. **text**)
      if (/^\d+\.\s+\*\*/.test(line)) {
        const match = line.match(/^(\d+)\.\s+\*\*(.+?)\*\*(.+)$/);
        if (match) {
          return (
            <div key={idx} style={{ marginBottom: '5px', paddingLeft: '10px' }}>
              <strong>{match[1]}. {match[2]}</strong>{match[3]}
            </div>
          );
        }
      }
      // Format bold text (**text**)
      if (line.includes('**')) {
        const parts = line.split(/(\*\*.+?\*\*)/g);
        return (
          <div key={idx} style={{ marginBottom: '5px' }}>
            {parts.map((part, pIdx) => 
              part.startsWith('**') && part.endsWith('**') ? (
                <strong key={pIdx}>{part.slice(2, -2)}</strong>
              ) : (
                <span key={pIdx}>{part}</span>
              )
            )}
          </div>
        );
      }
      // Regular line
      return <div key={idx} style={{ marginBottom: '5px' }}>{line || '\u00A0'}</div>;
    });
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <div style={{ marginBottom: '30px' }}>
        <h1 style={{ margin: 0, fontSize: '2.5rem', color: '#1976d2' }}>🤖 AdvisoryAI Jarvis</h1>
        <p style={{ margin: '5px 0', color: '#666', fontSize: '1.1rem' }}>Proactive Assistant for Financial Advisors</p>
      </div>

      <div style={{ display: 'flex', gap: '20px', marginTop: '20px', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 600px', minWidth: '400px' }}>
          <h2 style={{ marginTop: 0 }}>💬 Chat with Jarvis</h2>
          
          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '8px', 
            padding: '20px', 
            minHeight: '400px',
            maxHeight: '600px',
            marginBottom: '20px',
            overflowY: 'auto',
            backgroundColor: '#fafafa'
          }}>
            {messages.map((msg, idx) => (
              <div key={idx} style={{ 
                marginBottom: '15px',
                padding: '12px 15px',
                backgroundColor: msg.role === 'user' ? '#e3f2fd' : '#ffffff',
                borderRadius: '8px',
                borderLeft: `4px solid ${msg.role === 'user' ? '#1976d2' : '#4caf50'}`,
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
              }}>
                <div style={{ 
                  fontWeight: 'bold', 
                  marginBottom: '8px',
                  color: msg.role === 'user' ? '#1976d2' : '#4caf50',
                  fontSize: '14px'
                }}>
                  {msg.role === 'user' ? 'You' : 'Jarvis'}
                </div>
                <div style={{ 
                  marginTop: '5px', 
                  whiteSpace: 'pre-wrap',
                  lineHeight: '1.6',
                  color: '#333'
                }}>
                  {formatMessage(msg.content)}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ 
                padding: '12px', 
                color: '#666', 
                fontStyle: 'italic',
                textAlign: 'center'
              }}>
                ⏳ Thinking...
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !loading && sendMessage()}
              placeholder="Ask Jarvis anything... (e.g., 'What needs my attention today?')"
              style={{ 
                flex: 1, 
                padding: '12px 15px', 
                fontSize: '16px',
                border: '1px solid #ddd',
                borderRadius: '6px',
                outline: 'none'
              }}
              disabled={loading}
            />
            <button 
              onClick={sendMessage}
              disabled={loading || !message.trim()}
              style={{ 
                padding: '12px 24px', 
                fontSize: '16px',
                backgroundColor: loading || !message.trim() ? '#ccc' : '#1976d2',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: loading || !message.trim() ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              Send
            </button>
          </div>

          <div style={{ marginTop: '20px' }}>
            <button 
              onClick={loadBriefing} 
              disabled={loading}
              style={{
                padding: '12px 24px',
                fontSize: '16px',
                backgroundColor: loading ? '#ccc' : '#4caf50',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              📊 Load Daily Briefing
            </button>
          </div>
        </div>

        {briefing && (
          <div style={{ flex: '1 1 600px', minWidth: '400px' }}>
            <h2 style={{ marginTop: 0 }}>📋 Daily Briefing</h2>
            <div style={{ border: '1px solid #ccc', borderRadius: '8px', padding: '20px', maxHeight: '600px', overflowY: 'auto' }}>
              <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
                <h3 style={{ marginTop: 0 }}>Summary</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                  <div><strong>Reviews Due:</strong> {briefing.summary?.total_reviews_due || 0}</div>
                  <div><strong>Contact Gaps:</strong> {briefing.summary?.total_contact_gaps || 0}</div>
                  <div><strong>Milestones:</strong> {briefing.summary?.total_milestones || 0}</div>
                  <div><strong>Life Events:</strong> {briefing.summary?.total_life_events || 0}</div>
                  <div><strong>Concerns:</strong> {briefing.summary?.total_concerns || 0}</div>
                  <div><strong>Overdue Actions:</strong> {briefing.summary?.total_overdue_actions || 0}</div>
                  <div><strong>Overdue Follow-ups:</strong> {briefing.summary?.total_overdue_follow_ups || 0}</div>
                </div>
              </div>
              
              {briefing.reviews_due && briefing.reviews_due.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ color: '#d32f2f' }}>⚠️ Reviews Due ({briefing.reviews_due.length})</h4>
                  {briefing.reviews_due.slice(0, 10).map((review: any, idx: number) => (
                    <div key={idx} style={{ 
                      marginBottom: '10px', 
                      padding: '12px', 
                      backgroundColor: review.priority === 'high' ? '#ffebee' : '#fff3cd',
                      borderRadius: '6px',
                      borderLeft: `4px solid ${review.priority === 'high' ? '#d32f2f' : '#f57c00'}`
                    }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{review.client_name}</div>
                      <div style={{ fontSize: '14px', color: '#666' }}>
                        {review.days_overdue} days overdue (Last review: {review.last_review || 'N/A'})
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {briefing.contact_gaps && briefing.contact_gaps.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ color: '#f57c00' }}>📞 Contact Gaps ({briefing.contact_gaps.length})</h4>
                  {briefing.contact_gaps.slice(0, 10).map((gap: any, idx: number) => (
                    <div key={idx} style={{ 
                      marginBottom: '10px', 
                      padding: '12px', 
                      backgroundColor: '#fff3cd',
                      borderRadius: '6px'
                    }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{gap.client_name}</div>
                      <div style={{ fontSize: '14px', color: '#666' }}>
                        {gap.days_since_contact} days since last contact (Last: {gap.last_contact || 'N/A'})
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {briefing.upcoming_milestones && briefing.upcoming_milestones.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ color: '#1976d2' }}>🎂 Upcoming Milestones ({briefing.upcoming_milestones.length})</h4>
                  {briefing.upcoming_milestones.slice(0, 10).map((milestone: any, idx: number) => (
                    <div key={idx} style={{ 
                      marginBottom: '10px', 
                      padding: '12px', 
                      backgroundColor: '#e3f2fd',
                      borderRadius: '6px'
                    }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{milestone.client_name}</div>
                      <div style={{ fontSize: '14px', color: '#666' }}>
                        Turning {milestone.turning_age} in {milestone.days_until} days ({milestone.birthday_date})
                      </div>
                      {milestone.opportunity && (
                        <div style={{ fontSize: '13px', color: '#1976d2', marginTop: '5px', fontStyle: 'italic' }}>
                          {milestone.opportunity}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {briefing.life_events && briefing.life_events.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ color: '#7b1fa2' }}>🎯 Life Events ({briefing.life_events.length})</h4>
                  {briefing.life_events.slice(0, 10).map((event: any, idx: number) => (
                    <div key={idx} style={{ 
                      marginBottom: '10px', 
                      padding: '12px', 
                      backgroundColor: '#f3e5f5',
                      borderRadius: '6px'
                    }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{event.client_name}</div>
                      <div style={{ fontSize: '14px', color: '#666' }}>
                        {event.event_type} - {event.event_date}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {briefing.unresolved_concerns && briefing.unresolved_concerns.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ color: '#c62828' }}>⚠️ Unresolved Concerns ({briefing.unresolved_concerns.length})</h4>
                  {briefing.unresolved_concerns.slice(0, 10).map((concern: any, idx: number) => (
                    <div key={idx} style={{ 
                      marginBottom: '10px', 
                      padding: '12px', 
                      backgroundColor: '#ffebee',
                      borderRadius: '6px'
                    }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{concern.client_name}</div>
                      <div style={{ fontSize: '14px', color: '#666' }}>
                        {concern.concern} (Mentioned: {concern.first_mentioned})
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {briefing.overdue_actions && briefing.overdue_actions.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ color: '#d32f2f' }}>📋 Overdue Actions ({briefing.overdue_actions.length})</h4>
                  {briefing.overdue_actions.slice(0, 10).map((action: any, idx: number) => (
                    <div key={idx} style={{ 
                      marginBottom: '10px', 
                      padding: '12px', 
                      backgroundColor: '#ffebee',
                      borderRadius: '6px'
                    }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{action.client_name}</div>
                      <div style={{ fontSize: '14px', color: '#666' }}>
                        {action.action} - {action.days_overdue} days overdue (Due: {action.due_date})
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {briefing.overdue_follow_ups && briefing.overdue_follow_ups.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ color: '#f57c00' }}>📞 Overdue Follow-ups ({briefing.overdue_follow_ups.length})</h4>
                  {briefing.overdue_follow_ups.slice(0, 10).map((followup: any, idx: number) => (
                    <div key={idx} style={{ 
                      marginBottom: '10px', 
                      padding: '12px', 
                      backgroundColor: '#fff3cd',
                      borderRadius: '6px'
                    }}>
                      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{followup.client_name}</div>
                      <div style={{ fontSize: '14px', color: '#666' }}>
                        {followup.follow_up_type} - {followup.days_overdue} days overdue (Scheduled: {followup.scheduled_date})
                      </div>
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

