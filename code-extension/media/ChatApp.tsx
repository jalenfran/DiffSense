import React, { useState, useEffect, useRef } from 'react';

interface Message {
  id: string;
  text: string;
  from: 'user' | 'bot';
  timestamp: Date;
}

interface User {
  id: string;
  username: string;
  displayName?: string;
  photos?: Array<{ value: string }>;
}

interface ChatAppProps {
  vscode: any;
}

const ChatApp: React.FC<ChatAppProps> = ({ vscode }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  useEffect(() => {
    // Check authentication status on load
    vscode.postMessage({ command: 'checkAuth' });

    // Listen for messages from the extension
    const handleMessage = (event: MessageEvent) => {
      const msg = event.data;
      
      if (msg.command === 'newMessage') {
        setIsTyping(false);
        const newMessage: Message = {
          id: Date.now().toString(),
          text: msg.text,
          from: msg.from,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, newMessage]);
      } else if (msg.command === 'authStatus') {
        setIsAuthenticated(msg.authenticated);
        setUser(msg.user);
        setLoading(false);
      }
    };

    window.addEventListener('message', handleMessage);
    
    // Set a fallback timeout in case the auth check doesn't respond
    const fallbackTimeout = setTimeout(() => {
      if (loading) {
        setLoading(false);
        setIsAuthenticated(false);
      }
    }, 5000);

    return () => {
      window.removeEventListener('message', handleMessage);
      clearTimeout(fallbackTimeout);
    };
  }, [loading]);

  const handleSendMessage = () => {
    const text = inputValue.trim();
    if (!text) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      from: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);
    
    // Send message to extension
    vscode.postMessage({ command: 'sendMessage', text });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  const handleGitHubLogin = () => {
    vscode.postMessage({ command: 'githubLogin' });
  };

  const handleRefreshAuth = () => {
    setLoading(true);
    vscode.postMessage({ command: 'checkAuth' });
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: false 
    });
  };

  if (loading) {
    return (
      <div className="chat-container">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading DiffSense...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="chat-container">
        <div className="login-container">
          <div className="login-content">
            <div className="login-header">
              <div className="logo">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="logo-icon">
                  <path d="M7 6V3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3h3a1 1 0 0 1 1 1v4a1 1 0 0 1-1 1h-3v3a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1v-3H4a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h3z"/>
                </svg>
              </div>
              <h1>DiffSense</h1>
              <p>Analyze Your Git Diffs Like Never Before</p>
            </div>
            
            <div className="login-description">
              <p>Connect your GitHub account to start analyzing your repositories with AI-powered insights.</p>
            </div>            <button onClick={handleGitHubLogin} className="github-login-btn">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              Continue with GitHub
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" className="arrow-icon">
                <path d="M5 12h14m-7-7l7 7-7 7"/>
              </svg>
            </button>
            
            <div className="refresh-section">
              <p>Already signed in?</p>
              <button onClick={handleRefreshAuth} className="refresh-btn">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                </svg>
                Refresh
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="header-left">
          <div className="header-title">DiffSense AI</div>
          <div className="header-status">
            <span className="status-indicator"></span>
            Online
          </div>
        </div>        <div className="header-right">
          <div className="user-info">
            <div className="user-avatar-placeholder">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
              </svg>
            </div>
            <span className="username">@{user?.username}</span>
          </div>
        </div>
      </div>
      
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-icon">🤖</div>
            <h3>Welcome to DiffSense AI</h3>
            <p>I'm here to help you analyze your code changes and repository insights. Ask me anything about your Git diffs!</p>
          </div>
        )}
        
        {messages.map((message) => (
          <div 
            key={message.id} 
            className={`message ${message.from === 'user' ? 'user-message' : 'bot-message'}`}
          >
            <div className="message-content">
              <div className="message-text">{message.text}</div>
              <div className="message-time">{formatTime(message.timestamp)}</div>
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="message bot-message typing-indicator">
            <div className="message-content">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your code changes..."
            className="message-input"
            rows={1}
          />
          <button 
            onClick={handleSendMessage}
            disabled={!inputValue.trim()}
            className="send-button"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatApp;
