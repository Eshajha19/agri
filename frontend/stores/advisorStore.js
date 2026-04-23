import { create } from 'zustand';

export const useAdvisorStore = create((set) => ({
  // Advisor stats counters
  farmers: 0,
  setFarmers: (count) => set({ farmers: count }),

  crops: 0,
  setCrops: (count) => set({ crops: count }),

  languages: 0,
  setLanguages: (count) => set({ languages: count }),

  // Modal visibility
  showWeather: false,
  setShowWeather: (show) => set({ showWeather: show }),

  showSoilChatbot: false,
  setShowSoilChatbot: (show) => set({ showSoilChatbot: show }),

  showComingSoon: false,
  setShowComingSoon: (show) => set({ showComingSoon: show }),

  showIrrigation: false,
  setShowIrrigation: (show) => set({ showIrrigation: show }),

  showProfitCalculator: false,
  setShowProfitCalculator: (show) => set({ showProfitCalculator: show }),

  showFarmingMap: false,
  setShowFarmingMap: (show) => set({ showFarmingMap: show }),

  showCropDiseaseDetection: false,
  setShowCropDiseaseDetection: (show) => set({ showCropDiseaseDetection: show }),

  // Reset store
  resetAdvisorStore: () =>
    set({
      farmers: 0,
      crops: 0,
      languages: 0,
      showWeather: false,
      showSoilChatbot: false,
      showIrrigation: false,
      showProfitCalculator: false,
      showFarmingMap: false,
      showCropDiseaseDetection: false,
      showComingSoon: false,
    }),
}));
