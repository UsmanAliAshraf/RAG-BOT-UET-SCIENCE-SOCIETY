import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [showMemory, setShowMemory] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const addMessage = (sender, text, memoryInfo = null) => {
    setMessages((prev) => [...prev, { sender, text, memoryInfo, id: Date.now() }]);
  };

  const typeWriterEffect = async (text, memoryInfo = null) => {
    let currentText = "";
    const messageId = Date.now();
    
    // Add the bot message with empty text first
    setMessages((prev) => [...prev, { sender: "bot", text: "", memoryInfo, id: messageId }]);
    
    for (let char of text) {
      currentText += char;
      setMessages((prev) => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, text: currentText }
            : msg
        )
      );
      await new Promise((r) => setTimeout(r, 15));
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);
    
    // Add user message first
    addMessage("user", userMessage);
    
    // Add typing indicator
    const typingId = Date.now();
    setMessages((prev) => [...prev, { sender: "typing", text: "", id: typingId }]);

    try {
      const requestBody = {
        question: userMessage,
        session_id: sessionId
      };

      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      
      if (!sessionId && data.session_id) {
        setSessionId(data.session_id);
      }
      
      const memoryInfo = {
        content: data.memory_content,
        bufferLength: data.memory_buffer_length
      };
      
      // Remove typing indicator
      setMessages((prev) => prev.filter(msg => msg.id !== typingId));
      
      // Clean the answer one more time as a safety measure
      const cleanAnswer = cleanMemoryContent(data.answer);
      
      // Start typing effect
      await typeWriterEffect(cleanAnswer, memoryInfo);
    } catch (err) {
      console.error("Error:", err);
      // Remove typing indicator and add error message
      setMessages((prev) => prev.filter(msg => msg.id !== typingId));
      addMessage("bot", "❌ Error getting response. Please try again.");
    }
    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
    setShowMemory(false);
  };

  const cleanMemoryContent = (content) => {
    if (!content) return "No memory yet";
    
    // Comprehensive cleaning function to remove all think tag variations
    let cleaned = content;
    
    // Remove <think>...</think> blocks (case insensitive, multiline)
    cleaned = cleaned.replace(/<think>.*?<\/think>/gis, '');
    
    // Remove <thinking>...</thinking> blocks
    cleaned = cleaned.replace(/<thinking>.*?<\/thinking>/gis, '');
    
    // Remove [think]...[/think] blocks
    cleaned = cleaned.replace(/\[think\].*?\[\/think\]/gis, '');
    
    // Remove any remaining think-related patterns
    cleaned = cleaned.replace(/<think.*?>.*?<\/think>/gis, '');
    
    // Remove any lines that start with "think:" or "thinking:"
    cleaned = cleaned.replace(/^(think|thinking):.*$/gim, '');
    
    // Clean up extra whitespace and empty lines
    cleaned = cleaned.replace(/\n\s*\n/g, '\n').trim();
    
    return cleaned || "No memory yet";
  };

  const toggleMemory = () => {
    setShowMemory(!showMemory);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1> Echo - UET Science Society</h1>
        <div className="header-buttons">
          <button onClick={toggleMemory} className="memory-btn">
            {showMemory ? "🔒 Hide Memory" : "🧠 Show Memory"}
          </button>
          <button onClick={clearChat} className="clear-btn">
            🗑️ Clear Chat
          </button>
        </div>
      </div>
      
      <div className="chat-body">
        {messages.length === 0 && (
          <div className="welcome-message">
            <p>👋 Ask me anything.</p>
            <p>About the  Science Society their events, or activities!</p>
          </div>
        )}
        
        {messages.map((msg) => (
          <div key={msg.id}>
            {msg.sender === "typing" ? (
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            ) : (
              <div className={`chat-message ${msg.sender === "user" ? "user" : "bot"}`}>
                {msg.sender === "bot" ? (
                  <ReactMarkdown 
                    components={{
                      // Custom styling for markdown elements
                      h1: ({children}) => <h1 style={{fontSize: '1.2rem', margin: '8px 0', color: 'inherit'}}>{children}</h1>,
                      h2: ({children}) => <h2 style={{fontSize: '1.1rem', margin: '6px 0', color: 'inherit'}}>{children}</h2>,
                      h3: ({children}) => <h3 style={{fontSize: '1rem', margin: '4px 0', color: 'inherit'}}>{children}</h3>,
                      p: ({children}) => <p style={{margin: '4px 0', lineHeight: '1.5'}}>{children}</p>,
                      ul: ({children}) => <ul style={{margin: '8px 0', paddingLeft: '20px'}}>{children}</ul>,
                      ol: ({children}) => <ol style={{margin: '8px 0', paddingLeft: '20px'}}>{children}</ol>,
                      li: ({children}) => <li style={{margin: '2px 0', lineHeight: '1.4'}}>{children}</li>,
                      strong: ({children}) => <strong style={{fontWeight: '600', color: '#a855f7'}}>{children}</strong>,
                      em: ({children}) => <em style={{fontStyle: 'italic', color: '#d8b4fe'}}>{children}</em>,
                      code: ({children}) => <code style={{backgroundColor: 'rgba(168, 85, 247, 0.2)', padding: '2px 4px', borderRadius: '4px', fontSize: '0.9em'}}>{children}</code>,
                      a: ({href, children}) => <a href={href} target="_blank" rel="noopener noreferrer" style={{color: '#a855f7', textDecoration: 'underline'}}>{children}</a>,
                    }}
                  >
                    {msg.text}
                  </ReactMarkdown>
                ) : (
                  msg.text
                )}
              </div>
            )}
            
            {showMemory && msg.sender === "bot" && msg.memoryInfo && (
              <div className="memory-display">
                <div className="memory-header">
                  🧠 Memory (Buffer: {msg.memoryInfo.bufferLength} messages)
                </div>
                <div className="memory-content">
                  <ReactMarkdown 
                    components={{
                      p: ({children}) => <p style={{margin: '2px 0', fontSize: '0.85em', lineHeight: '1.3'}}>{children}</p>,
                      ul: ({children}) => <ul style={{margin: '4px 0', paddingLeft: '15px', fontSize: '0.85em'}}>{children}</ul>,
                      li: ({children}) => <li style={{margin: '1px 0', fontSize: '0.85em'}}>{children}</li>,
                      strong: ({children}) => <strong style={{fontWeight: '600', color: '#a855f7', fontSize: '0.85em'}}>{children}</strong>,
                    }}
                  >
                    {cleanMemoryContent(msg.memoryInfo.content)}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      
      <div className="chat-input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={loading}
          rows={1}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? "⏳" : "Send"}
        </button>
      </div>
      
      {sessionId && (
        <div className="session-info">
          <small>Session: {sessionId.substring(0, 8)}...</small>
        </div>
      )}
    </div>
  );
};

export default App;
