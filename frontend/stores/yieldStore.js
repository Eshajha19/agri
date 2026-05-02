import { create } from 'zustand';
import { predictYield } from '../services/yieldApi';

export const useYieldStore = create((set, get) => ({
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
  yieldLastUpdated: null,
  setYieldPrediction: (prediction) => set({ yieldPrediction: prediction, yieldLastUpdated: Date.now() }),

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
      const data = await predictYield(get().yieldForm);
      set({ yieldPrediction: data.predicted_ExpYield, yieldLastUpdated: Date.now(), showYieldPopup: true });
    } catch (error) {
      const message =
        error?.response?.data?.detail ||
        error.message ||
        'Failed to get prediction';
      set({ yieldError: message });
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
      yieldLastUpdated: null,
      yieldError: null,
      showYieldPopup: false,
    }),
}));
