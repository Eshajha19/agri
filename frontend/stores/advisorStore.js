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

  showPestManagement: false,
  setShowPestManagement: (show) => set({ showPestManagement: show }),

  showSoilAnalysis: false,
  setShowSoilAnalysis: (show) => set({ showSoilAnalysis: show }),

  showSoilGuide: false,
  setShowSoilGuide: (show) => set({ showSoilGuide: show }),

   showFertilizerPopup: false,
   setShowFertilizerPopup: (show) => set({ showFertilizerPopup: show }),

  showFarmDiary: false,
  setShowFarmDiary: (show) => set({ showFarmDiary: show }),

  showAgriMarketplace: false,
  setShowAgriMarketplace: (show) => set({ showAgriMarketplace: show }),

  showAgriLMS: false,
  setShowAgriLMS: (show) => set({ showAgriLMS: show }),

  showQRTraceability: false,
  setShowQRTraceability: (show) => set({ showQRTraceability: show }),

  // Reset all modals to closed
  resetAdvisorStore: () =>
    set({
      farmers: 0,
      crops: 0,
      languages: 0,
      showWeather: false,
      showSoilChatbot: false,
      showSoilAnalysis: false,
      showSoilGuide: false,
      showIrrigation: false,
      showProfitCalculator: false,
      showFertilizerPopup: false,
      showFarmingMap: false,
      showCropDiseaseDetection: false,
      showPestManagement: false,
      showComingSoon: false,
      showAgriMarketplace: false,
      showAgriLMS: false,
      showQRTraceability: false,
      showFarmDiary: false,
    }),
}));
