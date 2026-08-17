import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { parseVoiceNavigation } from './lib/voiceNavigation';
import './VoiceMicWidget.css';

const VoiceMicWidget = () => {
  const navigate = useNavigate();
  const [isListening, setIsListening] = useState(false);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const recognitionRef = useRef(null);
  const statusTimeoutRef = useRef(null);

  const clearStatusTimeout = useCallback(() => {
    if (statusTimeoutRef.current) {
      clearTimeout(statusTimeoutRef.current);
      statusTimeoutRef.current = null;
    }
  }, []);

  const showStatus = useCallback((message, duration = 2000) => {
    clearStatusTimeout();
    setStatus(message);
    statusTimeoutRef.current = setTimeout(() => {
      setStatus('idle');
    }, duration);
  }, [clearStatusTimeout]);

  const statusRef = useRef(status);
  statusRef.current = status;

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Speech Recognition not supported in this browser');
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = false;
    recognitionRef.current.lang = 'en-US';

    recognitionRef.current.onstart = () => {
      setIsListening(true);
      setError(null);
      showStatus('listening');
    };

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      showStatus('processing', 1500);

      const nav = parseVoiceNavigation(transcript);
      if (nav.type === 'back') {
        navigate(-1);
        showStatus('Going back...');
      } else if (nav.type === 'navigate') {
        navigate(nav.path);
        showStatus(`Opening ${nav.path.replace('/', '').replace('-', ' ')}`);
      } else {
        showStatus('Command not recognized');
      }
    };

    recognitionRef.current.onerror = (event) => {
      setIsListening(false);
      if (event.error === 'not-allowed') {
        setError('Microphone access denied');
        showStatus('Mic blocked', 3000);
      } else if (event.error === 'no-speech') {
        showStatus('No speech detected');
      } else {
        showStatus('Error. Try again.');
      }
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
      showStatus('idle');
    };

    return () => {
      clearStatusTimeout();
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {
          // ignore
        }
      }
    };
  }, [navigate, showStatus, clearStatusTimeout]);

  const toggleListening = () => {
    if (!recognitionRef.current) {
      setError('Speech Recognition not available');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      showStatus('idle');
    } else {
      setError(null);
      try {
        recognitionRef.current.start();
      } catch {
        showStatus('Failed to start');
      }
    }
  };

  return (
    <div className="voice-mic-widget">
      <button
        type="button"
        onClick={toggleListening}
        className={`voice-mic-button ${isListening ? 'listening' : ''}`}
        aria-label={isListening ? 'Stop listening' : 'Start voice navigation'}
        title={isListening ? 'Listening... Tap to stop' : 'Tap to speak a command'}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      </button>

      {isListening && <span className="voice-mic-pulse" />}

      {status !== 'idle' && (
        <div className={`voice-mic-status ${status}`}>
          {status === 'listening' && 'Listening...'}
          {status === 'processing' && 'Processing...'}
          {status === 'Going back...' && 'Going back...'}
          {status.startsWith('Opening') && status}
          {status === 'Command not recognized' && 'Try: "open dashboard"'}
          {status === 'Mic blocked' && 'Mic blocked'}
          {status === 'No speech detected' && 'No speech detected'}
          {status === 'Failed to start' && 'Failed to start'}
          {status === 'Error. Try again.' && 'Error. Try again.'}
        </div>
      )}

      {error && (
        <div className="voice-mic-error">
          {error}
        </div>
      )}
    </div>
  );
};

export default VoiceMicWidget;
