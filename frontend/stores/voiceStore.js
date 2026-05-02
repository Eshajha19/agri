import { create } from 'zustand';

export const useVoiceStore = create((set) => ({
  isListening: false,
  isSpeaking: false,
  isLoading: false,
  transcript: '',
  language: 'hi-IN',
  messages: [],

  setIsListening: (val) => set({ isListening: val }),
  setIsSpeaking: (val) => set({ isSpeaking: val }),
  setIsLoading: (val) => set({ isLoading: val }),
  setTranscript: (val) => set({ transcript: val }),
  setLanguage: (val) => set({ language: val }),
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
  clearMessages: () => set({ messages: [] }),
  resetVoiceStore: () =>
    set({
      isListening: false,
      isSpeaking: false,
      isLoading: false,
      transcript: '',
      messages: [],
    }),
}));
