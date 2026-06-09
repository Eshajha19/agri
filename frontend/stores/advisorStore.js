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

  showOfflineStatus: false,
  setShowOfflineStatus: (show) => set({ showOfflineStatus: show }),

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
  showSprayReminder: false,
  setShowSprayReminder: (show) => set({ showSprayReminder: show }),
  showSprayScheduler: false,
  setShowSprayScheduler: (show) => set({ showSprayScheduler: show }),
  showPestCalendar: false,
  setShowPestCalendar: (show) => set({ showPestCalendar: show }),
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

    showCropRecommendationAdvisor: false,
    setShowCropRecommendationAdvisor: (show) => set({ showCropRecommendationAdvisor: show }),

    showSeedVerifier: false,
    setShowSeedVerifier: (show) => set({ showSeedVerifier: show }),

   showGeoAlerts: false,
   setShowGeoAlerts: (show) => set({ showGeoAlerts: show }),

   showClimateSimulator: false,
   setShowClimateSimulator: (show) => set({ showClimateSimulator: show }),

   showRAGAdvisor: false,
   setShowRAGAdvisor: (show) => set({ showRAGAdvisor: show }),

showGreenPractices: false,
     setShowGreenPractices: (show) => set({ showGreenPractices: show }),

     showEquipmentManagement: false,
     setShowEquipmentManagement: (show) => set({ showEquipmentManagement: show }),

     showCropGrading: false,
     setShowCropGrading: (show) => set({ showCropGrading: show }),

   showSustainabilityAnalytics: false,
   setShowSustainabilityAnalytics: (show) => set({ showSustainabilityAnalytics: show }),
   showExpertDirectory: false,
   setShowExpertDirectory: (show) => set({ showExpertDirectory: show }),

   showTeleConsultation: false,
   setShowTeleConsultation: (show) => set({ showTeleConsultation: show }),

   activeConsultation: null,
   setActiveConsultation: (consultation) => set({ activeConsultation: consultation }),

showConsultationHistory: false,
    setShowConsultationHistory: (show) => set({ showConsultationHistory: show }),

    showCropInsuranceClaim: false,
    setShowCropInsuranceClaim: (show) => set({ showCropInsuranceClaim: show }),

    selectedExpert: null,
   setSelectedExpert: (expert) => set({ selectedExpert: expert }),

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
         showSprayReminder: false,
         showSprayScheduler: false,
        showPestCalendar: false,
       showOfflineStatus: false,
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
         showGreenPractices: false,
         showEquipmentManagement: false,
          showCropGrading: false,
showSustainabilityAnalytics: false,
           showExpertDirectory: false,
           showTeleConsultation: false,
           activeConsultation: null,
           showConsultationHistory: false,
           selectedExpert: null,
           showCropInsuranceClaim: false,
       }),
}));
