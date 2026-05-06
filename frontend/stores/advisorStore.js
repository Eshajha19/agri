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

  showFarmPlanner3D: false,
  setShowFarmPlanner3D: (show) => set({ showFarmPlanner3D: show }),

  showCropRotation: false,
  setShowCropRotation: (show) => set({ showCropRotation: show }),

   showForecast: false,
   setShowForecast: (show) => set({ showForecast: show }),

   showExpertStatus: false,
   setShowExpertStatus: (show) => set({ showExpertStatus: show }),

   showBankReport: false,
   setShowBankReport: (show) => set({ showBankReport: show }),

   showP2PChat: false,
   setShowP2PChat: (show) => set({ showP2PChat: show }),

   showSmartCropRecommendation: false,
   setShowSmartCropRecommendation: (show) => set({ showSmartCropRecommendation: show }),

   showSeedVerifier: false,
   setShowSeedVerifier: (show) => set({ showSeedVerifier: show }),

   showGeoAlerts: false,
   setShowGeoAlerts: (show) => set({ showGeoAlerts: show }),

   showClimateSimulator: false,
   setShowClimateSimulator: (show) => set({ showClimateSimulator: show }),

   showRAGAdvisor: false,
   setShowRAGAdvisor: (show) => set({ showRAGAdvisor: show }),

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
       showQRTraceability: false,
       showFarmPlanner3D: false,
       showFarmDiary: false,
       showAgriLMS: false,
       showForecast: false,
       showExpertStatus: false,
       showBankReport: false,
       showCropRotation: false,
       showP2PChat: false,
       showSmartCropRecommendation: false,
       showSeedVerifier: false,
       showGeoAlerts: false,
       showClimateSimulator: false,
       showRAGAdvisor: false,
     }),
}));
