import { describe, it, expect } from 'vitest';
import { getCropRecommendations, getYearlyCycle } from '../utils/cropPlanner';

describe('cropPlanner', () => {
  describe('getCropRecommendations', () => {
    it('should recommend Paddy for Kharif season in Alluvial soil in West Bengal', () => {
      const recommendations = getCropRecommendations('West Bengal', 'Kharif', 'Alluvial');
      
      expect(recommendations.length).toBeGreaterThan(0);
      
      // Paddy should be one of the top results (score >= 4)
      const paddy = recommendations.find(c => c.name === 'Paddy (Rice)');
      expect(paddy).toBeDefined();
      expect(paddy.score).toBeGreaterThanOrEqual(4);
    });

    it('should include Annual crops like Sugarcane regardless of season if soil/region matches', () => {
      const recommendations = getCropRecommendations('Uttar Pradesh', 'Rabi', 'Alluvial');
      
      const sugarcane = recommendations.find(c => c.name === 'Sugarcane');
      expect(sugarcane).toBeDefined();
    });

    it('should not recommend Rabi crops during Zaid season unless they span both', () => {
      const recommendations = getCropRecommendations('Punjab', 'Zaid', 'Sandy');
      
      // Wheat is strictly Rabi
      const wheat = recommendations.find(c => c.name === 'Wheat');
      expect(wheat).toBeUndefined();
    });
  });

  describe('getYearlyCycle', () => {
    it('should return a cycle containing kharif, rabi, and zaid arrays', () => {
      const cycle = getYearlyCycle('Maharashtra', 'Black');
      
      expect(cycle).toHaveProperty('kharif');
      expect(cycle).toHaveProperty('rabi');
      expect(cycle).toHaveProperty('zaid');
      
      expect(Array.isArray(cycle.kharif)).toBe(true);
      expect(Array.isArray(cycle.rabi)).toBe(true);
      expect(Array.isArray(cycle.zaid)).toBe(true);
    });
  });
});
