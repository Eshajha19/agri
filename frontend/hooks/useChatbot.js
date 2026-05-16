import { useCallback, useRef, useEffect } from 'react';
import { useChatbotStore } from '../stores/chatbotStore';
import { useErrorHandler } from './useErrorHandler';
import apiClient from '../services/api';

export const useChatbot = () => {
  const { handleWarning } = useErrorHandler();
  const geminiApiKey = import.meta.env.VITE_GEMINI_API_KEY || import.meta.env.VITE_API_KEY;
  const isGeminiConfigured = Boolean(geminiApiKey);
  const {
    messages,
    addMessage,
    setMessages,
    userInput,
    setUserInput,
    soilImage,
    setSoilImage,
    isListening,
    setIsListening,
    isSpeaking,
    setIsSpeaking,
    isLoading,
    setIsLoading,
    resetChatbotStore,
  } = useChatbotStore();

  const recognitionRef = useRef(null);
  const utteranceRef = useRef(null);

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setUserInput(transcript);
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event) => {
        handleWarning(`Speech recognition: ${event.error}`, 'speech-recognition');
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, [setUserInput, setIsListening]);

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true);
      recognitionRef.current.start();
    }
  }, [isListening, setIsListening]);

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
        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        utteranceRef.current = utterance;
        setIsSpeaking(true);
        window.speechSynthesis.speak(utterance);
      }
    },
    [setIsSpeaking]
  );

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, [setIsSpeaking]);

  const toBase64 = useCallback(
    (file) =>
      new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = (error) => reject(error);
      }),
    []
  );

  const handleImageUpload = useCallback(
    (file) => {
      setSoilImage(file);
    },
    [setSoilImage]
  );

  const handleSendMessage = useCallback(
    async (text, imageFile) => {
      if (!text.trim() && !imageFile) return;

      // Add user message
      addMessage({ text: text || '(Image uploaded)', from: 'user' });
      setUserInput('');
      setSoilImage(null);

      setIsLoading(true);
      try {
        const parts = [];
        const systemPrompt =
          "You are an AI Agricultural Expert. Provide helpful suggestions for crops, weather-based farming advice, and soil health. Keep answers concise and practical for farmers. Mention climate conditions specifically if relevant.";

        parts.push({ text: `${systemPrompt}\n\nUser Question: ${text}` });

        if (imageFile) {
          parts.push({
            inline_data: {
              data: await toBase64(imageFile),
              mime_type: imageFile.type,
            },
          });
        }

        if (!isGeminiConfigured) {
          setIsLoading(false);
          addMessage({
            text: 'AI chat is temporarily unavailable right now. Please use the crop guide, weather alerts, or pest tools for support.',
            from: 'bot',
          });
          return;
        }

        const response = await apiClient.post(
          'https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent',
          {
            contents: [{ parts }],
          },
          {
            params: { key: geminiApiKey },
            retries: 1,
            errorContext: 'chatbot-message',
            errorMessage: 'Failed to process your message. Please try again.',
          }
        );

        const data = response.data;

        if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
          const botMessage = data.candidates[0].content.parts[0].text;
          addMessage({ text: botMessage, from: 'bot' });
          speak(botMessage);
        } else {
          addMessage({
            text: 'Could not process your request. Please try again.',
            from: 'bot',
          });
        }
      } catch {
        addMessage({
          text: 'An error occurred. Please try again later.',
          from: 'bot',
        });
      } finally {
        setIsLoading(false);
      }
    },
    [addMessage, setUserInput, setSoilImage, setIsLoading, toBase64, speak]
  );

  return {
    messages,
    addMessage,
    setMessages,
    userInput,
    setUserInput,
    soilImage,
    setSoilImage: handleImageUpload,
    isListening,
    startListening,
    stopListening,
    toggleListening,
    isSpeaking,
    speak,
    stopSpeaking,
    isLoading,
    handleSendMessage,
    isGeminiConfigured,
    resetChatbotStore,
  };
};
