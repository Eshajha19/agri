import { create } from 'zustand';

export const useYieldStore = create((set) => ({
  // Yield form state
  yieldForm: {
    Crop: 'Paddy',
    CropCoveredArea: 50,
    CHeight: 50,
    CNext: 'Lentil',
    CLast: 'Pea',
    CTransp: 'Transplanting',
    IrriType: 'Flood',
    IrriSource: 'Groundwater',
    IrriCount: 3,
    WaterCov: 50,
    Season: 'Rabi',
  },

  // Update specific form field
  updateYieldFormField: (field, value) =>
    set((state) => ({
      yieldForm: { ...state.yieldForm, [field]: value },
    })),

  // Update entire form
  setYieldForm: (form) => set({ yieldForm: form }),

  // Prediction results
  yieldPrediction: null,
  setYieldPrediction: (prediction) => set({ yieldPrediction: prediction }),

  // Error handling
  yieldError: null,
  setYieldError: (error) => set({ yieldError: error }),

  // Loading state
  yieldLoading: false,
  setYieldLoading: (loading) => set({ yieldLoading: loading }),

  // Popup visibility
  showYieldPopup: false,
  setShowYieldPopup: (show) => set({ showYieldPopup: show }),

  // Fetch yield prediction
  fetchYield: async function () {
    set({ yieldLoading: true, yieldError: null });
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.yieldForm),
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const data = await response.json();
      set({ yieldPrediction: data.predicted_ExpYield, showYieldPopup: true });
    } catch (error) {
      console.error('Error fetching yield:', error);
      set({ yieldError: error.message || 'Failed to get prediction' });
    } finally {
      set({ yieldLoading: false });
    }
  },

  // Reset store
  resetYieldStore: () =>
    set({
      yieldForm: {
        Crop: 'Paddy',
        CropCoveredArea: 50,
        CHeight: 50,
        CNext: 'Lentil',
        CLast: 'Pea',
        CTransp: 'Transplanting',
        IrriType: 'Flood',
        IrriSource: 'Groundwater',
        IrriCount: 3,
        WaterCov: 50,
        Season: 'Rabi',
      },
      yieldPrediction: null,
      yieldError: null,
      showYieldPopup: false,
    }),
}));
