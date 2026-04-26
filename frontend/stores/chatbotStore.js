import { create } from 'zustand';

export const useChatbotStore = create((set) => ({
  // Chat messages
  messages: [
    {
      text: "Hello! I'm your Agricultural Assistant. How can I help you today?",
      from: 'bot',
    },
  ],
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  setMessages: (messages) => set({ messages }),

  // User input
  userInput: '',
  setUserInput: (input) => set({ userInput: input }),

  // Soil image
  soilImage: null,
  setSoilImage: (image) => set({ soilImage: image }),

  // Speech recognition
  isListening: false,
  setIsListening: (listening) => set({ isListening: listening }),

  // Text-to-speech
  isSpeaking: false,
  setIsSpeaking: (speaking) => set({ isSpeaking: speaking }),

  // Loading state for API calls
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),

  // Reset store
  resetChatbotStore: () =>
    set({
      messages: [
        {
          text: "Hello! I'm your Agricultural Assistant. How can I help you today?",
          from: 'bot',
        },
      ],
      userInput: '',
      soilImage: null,
      isListening: false,
      isSpeaking: false,
      isLoading: false,
    }),
}));
