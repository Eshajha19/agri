import React, { useState, useRef, useEffect } from 'react';
import './VoiceAssistant.css';
import apiClient from './lib/apiClient';

const VoiceAssistant = () => {
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState('hi');
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState(null);
  const [inputText, setInputText] = useState('');
  const [showTranscript, setShowTranscript] = useState(false);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioContextRef = useRef(null);

  // Language labels
  const languages = {
    hi: { name: 'Hindi', label: 'हिंदी' },
    bho: { name: 'Bhojpuri', label: 'भोजपुरी' },
    mr: { name: 'Marathi', label: 'मराठी' },
    gu: { name: 'Gujarati', label: 'ગુજરાતી' },
    kn: { name: 'Kannada', label: 'ಕನ್ನಡ' },
    te: { name: 'Telugu', label: 'తెలుగు' },
    ta: { name: 'Tamil', label: 'தமிழ்' },
    en: { name: 'English', label: 'English' },
  };

  // Initialize session on mount
  useEffect(() => {
    createSession();
    setupSpeechRecognition();
  }, []);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const createSession = async () => {
    try {
      const response = await apiClient.post('/api/voice/sessions/create', {
        language_code: language,
      });

      setSessionId(response.data.session_id);
    } catch (err) {
      console.error('Session creation error:', err);
      setError('Failed to create voice session');
    }
  };

  const setupSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Speech Recognition not supported in this browser');
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;

    recognitionRef.current.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognitionRef.current.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      if (finalTranscript) {
        setInputText(finalTranscript.trim());
        setShowTranscript(false);
      } else if (interimTranscript) {
        setShowTranscript(true);
      }
    };

    recognitionRef.current.onerror = (event) => {
      setError(`Speech recognition error: ${event.error}`);
      setIsListening(false);
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
    };
  };

  const toggleListening = () => {
    if (!recognitionRef.current) {
      setError('Speech Recognition not available');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.lang = language;
      recognitionRef.current.start();
    }
  };

  const processQuery = async (query) => {
    if (!query.trim() || !sessionId) {
      setError('Please provide a query and ensure session is active');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Add user message to chat
      const userMessage = {
        id: `user_${Date.now()}`,
        type: 'user',
        text: query,
        language: language,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Send query to backend
      const response = await apiClient.post('/api/voice/query', {
        transcript: query,
        language_code: language,
        session_id: sessionId,
      });

      const data = response.data;

      // Add assistant response
      const assistantMessage = {
        id: `assistant_${Date.now()}`,
        type: 'assistant',
        text: data.response_text,
        intent: data.intent,
        language: data.language_code,
        offline: data.offline_mode,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Text-to-speech (optional)
      speakResponse(data.response_text, language);
    } catch (err) {
      console.error('Query processing error:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
      setInputText('');
    }
  };

  const speakResponse = (text, lang) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = `${lang}-IN`;
    utterance.rate = 0.9;
    window.speechSynthesis.cancel(); // Cancel any ongoing speech
    window.speechSynthesis.speak(utterance);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim()) {
      processQuery(inputText);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="voice-assistant-container">
      <div className="voice-assistant-header">
        <h1>🎤 Voice Assistant</h1>
        <p className="subtitle">Speak in your language • Get instant guidance</p>
      </div>

      {/* Status Bar */}
      <div className="voice-status-bar">
        <div className="status-item">
          <span className="status-label">Language:</span>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="language-select"
            disabled={isListening}
          >
            {Object.entries(languages).map(([code, { name, label }]) => (
              <option key={code} value={code}>
                {name} - {label}
              </option>
            ))}
          </select>
        </div>
        <div className="status-item">
          <span className={`status-badge ${isListening ? 'listening' : 'idle'}`}>
            {isListening ? '🎙️ Listening...' : '✓ Ready'}
          </span>
        </div>
        {sessionId && (
          <div className="status-item">
            <span className="session-id" title={sessionId}>
              Session: {sessionId.slice(0, 8)}...
            </span>
          </div>
        )}
      </div>

      {/* Messages Display */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">🌾</div>
            <h3>Welcome to Voice Assistant!</h3>
            <p>Ask me about:</p>
            <ul className="suggestions">
              <li>Crop health & diseases</li>
              <li>Fertilizer recommendations</li>
              <li>Irrigation schedules</li>
              <li>Weather alerts</li>
              <li>Yield predictions</li>
              <li>Pest management</li>
            </ul>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg) => (
              <div key={msg.id} className={`message message-${msg.type}`}>
                <div className="message-avatar">
                  {msg.type === 'user' ? '👨‍🌾' : '🤖'}
                </div>
                <div className="message-content">
                  <p>{msg.text}</p>
                  <div className="message-meta">
                    <span className="timestamp">{msg.timestamp}</span>
                    {msg.intent && (
                      <span className="intent-badge">{msg.intent.replace(/_/g, ' ')}</span>
                    )}
                    {msg.offline && (
                      <span className="offline-badge">⚡ Offline</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message message-assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)} className="close-error">×</button>
        </div>
      )}

      {/* Input Area */}
      <div className="voice-input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-wrapper">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={
                isListening
                  ? 'Listening... (tap mic to stop)'
                  : 'Type your question or tap 🎤'
              }
              className="text-input"
              disabled={isLoading}
            />
            {showTranscript && (
              <div className="transcript-preview">Listening...</div>
            )}
          </div>

          <div className="button-group">
            <button
              type="button"
              onClick={toggleListening}
              className={`mic-button ${isListening ? 'active' : ''}`}
              title={isListening ? 'Stop listening' : 'Start listening'}
              disabled={isLoading}
            >
              🎤
            </button>
            <button
              type="submit"
              className="send-button"
              disabled={!inputText.trim() || isLoading}
            >
              {isLoading ? '⏳' : '📤'}
            </button>
            {messages.length > 0 && (
              <button
                type="button"
                onClick={clearChat}
                className="clear-button"
                title="Clear chat history"
              >
                🗑️
              </button>
            )}
          </div>
        </form>
      </div>

      {/* Info Footer */}
      <div className="voice-footer">
        <p>💡 Tip: For best results, speak clearly and naturally in your regional language</p>
        <p>📱 Works offline with pre-loaded knowledge base</p>
      </div>
    </div>
  );
};

export default VoiceAssistant;
