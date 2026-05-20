// frontend/featureFlags/flagConfig.js

/**
 * Default feature flag definitions.
 * 
 * These serve as a safety fallback if the backend API is unreachable.
 * ML/RAG features should generally default to 'false' (disabled) here
 * to ensure a safe initial rollout.
 */
export const DEFAULT_FLAGS = {
  // ML/RAG Features
  rag_advisor_v2: {
    enabled: false,
    rollout_pct: 0,
    description: "RAG Advisor v2 with hybrid retrieval (experimental)"
  },
  yield_prediction_lstm: {
    enabled: false,
    rollout_pct: 0,
    description: "LSTM-based yield prediction model"
  },
  smart_crop_recommendation: {
    enabled: true,
    rollout_pct: 100,
    description: "AI-powered crop recommendation system"
  },
  climate_simulator_ml: {
    enabled: false,
    rollout_pct: 10,
    description: "ML-enhanced climate simulation (Beta)"
  },
  pest_detection_v2: {
    enabled: false,
    rollout_pct: 0,
    description: "Computer Vision based pest detection v2"
  },
  
  // UI / UX Features
  new_advisor_dashboard: {
    enabled: true,
    rollout_pct: 100,
    description: "Modernised Advisor Dashboard layout"
  },
  dark_mode_beta: {
    enabled: false,
    rollout_pct: 50,
    description: "Experimental high-contrast dark mode"
  },
  voice_assistant_kvk: {
    enabled: false,
    rollout_pct: 0,
    description: "Voice-activated KVK support"
  }
};
