import { useCallback, useRef, useEffect } from 'react';
import { useVoiceStore } from '../stores/voiceStore';
import { useErrorHandler } from './useErrorHandler';
import apiClient from '../services/api';

export const useVoiceAssistant = () => {
  const { handleWarning } = useErrorHandler();
  const {
    isListening,
    setIsListening,
    isSpeaking,
    setIsSpeaking,
    isLoading,
    setIsLoading,
    transcript,
    setTranscript,
    language,
    setLanguage,
    messages,
    addMessage,
    clearMessages,
    resetVoiceStore,
  } = useVoiceStore();

  const recognitionRef = useRef(null);

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = language;

      recognitionRef.current.onresult = (event) => {
        const text = event.results[0][0].transcript;
        setTranscript(text);
        setIsListening(false);
        handleVoiceQuery(text);
      };

      recognitionRef.current.onerror = (event) => {
        handleWarning(`Voice recognition error: ${event.error}`, 'voice-assistant');
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, [language]);

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.lang = language;
      setIsListening(true);
      recognitionRef.current.start();
    }
  }, [isListening, language, setIsListening]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [setIsListening]);

  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }, [isListening, startListening, stopListening]);

  const speak = useCallback(
    (text) => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = language;
        utterance.rate = 0.9;
        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        window.speechSynthesis.speak(utterance);
      }
    },
    [language, setIsSpeaking]
  );

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, [setIsSpeaking]);

  const handleVoiceQuery = useCallback(
    async (text) => {
      if (!text.trim()) return;

      addMessage({ text, from: 'user' });
      setIsLoading(true);

      try {
        const API_KEY = import.meta.env.VITE_API_KEY;

        if (!API_KEY) {
          setIsLoading(false);
          const errMsg =
            language === 'hi-IN'
              ? '⚠️ API Key nahi mili. Kripya Gemini API key add karein.'
              : '⚠️ API Key is missing. Please add your Gemini API key.';
          addMessage({ text: errMsg, from: 'bot' });
          speak(errMsg);
          return;
        }

        const systemPrompt =
          language === 'hi-IN'
            ? 'Aap ek AI Krishi Visheshagya hain. Kisan ke sawaalon ka jawab Hindi mein, simple aur practical tarike se do. Fasal, mausam, mitti aur kheti se related salah do. Jawab chhota aur seedha rakho.'
            : 'You are an AI Agricultural Expert. Provide helpful suggestions for crops, weather-based farming advice, and soil health. Keep answers concise and practical for farmers.';

        const response = await apiClient.post(
          'https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent',
          {
            contents: [
              {
                parts: [{ text: `${systemPrompt}\n\nUser Question: ${text}` }],
              },
            ],
          },
          {
            params: { key: API_KEY },
            retries: 1,
            errorContext: 'voice-assistant',
            errorMessage: 'Failed to process voice query. Please try again.',
          }
        );

        const data = response.data;

        if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
          const botMessage = data.candidates[0].content.parts[0].text;
          addMessage({ text: botMessage, from: 'bot' });
          speak(botMessage);
        } else {
          const fallback =
            language === 'hi-IN'
              ? 'Kripya dobara try karein.'
              : 'Could not process your request. Please try again.';
          addMessage({ text: fallback, from: 'bot' });
        }
      } catch {
        const errMsg =
          language === 'hi-IN'
            ? 'Kuch galat ho gaya. Baad mein try karein.'
            : 'An error occurred. Please try again later.';
        addMessage({ text: errMsg, from: 'bot' });
      } finally {
        setIsLoading(false);
      }
    },
    [addMessage, setIsLoading, speak, language]
  );

  return {
    isListening,
    isSpeaking,
    isLoading,
    transcript,
    language,
    setLanguage,
    messages,
    toggleListening,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    handleVoiceQuery,
    clearMessages,
    resetVoiceStore,
  };
};
