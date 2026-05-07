import { describe, it, expect } from 'vitest';
import { calculateSingleCropProfit, compareProfits } from '../utils/profitCalculator';

describe('profitCalculator', () => {
  describe('calculateSingleCropProfit', () => {
    it('should correctly calculate profit for the example scenario', () => {
      // Scenario: Wheat | Sandy Soil | Maharashtra
      // We'll simulate the economics: Cost: 20000, Yield: 25q, Price: 2600
      // Revenue: 25 * 2600 = 65000
      // Profit: 65000 - 20000 = 45000
      const result = calculateSingleCropProfit(20000, 25, 2600);
      expect(result.revenue).toBe(65000);
      expect(result.profit).toBe(45000);
      expect(result.profitPercentage).toBe('225.00'); // (45000/20000)*100
    });

    it('should calculate loss correctly', () => {
      const result = calculateSingleCropProfit(50000, 10, 2000);
      expect(result.revenue).toBe(20000);
      expect(result.profit).toBe(-30000);
      expect(result.profitPercentage).toBe('-60.00');
    });

    it('should handle zero inputs correctly', () => {
      const result = calculateSingleCropProfit(0, 0, 0);
      expect(result.profit).toBe(0);
      expect(result.profitPercentage).toBe('0.00');
    });

    it('should throw error on negative inputs', () => {
      expect(() => calculateSingleCropProfit(-10, 10, 10)).toThrow();
    });
  });

  describe('compareProfits', () => {
    it('should accurately compare multiple crops and identify the winner', () => {
      const crops = [
        { id: 1, cropName: 'Wheat', farmingCost: '20000', expectedYield: '25', marketPrice: '2600' }, // Profit 45000
        { id: 2, cropName: 'Paddy', farmingCost: '25000', expectedYield: '30', marketPrice: '2000' }, // Profit 35000
        { id: 3, cropName: 'Cotton', farmingCost: '30000', expectedYield: '15', marketPrice: '6000' } // Profit 60000
      ];

      const result = compareProfits(crops);
      expect(result.data.length).toBe(3);
      expect(result.bestCropIds).toEqual([3]);
    });

    it('should handle empty arrays', () => {
      const result = compareProfits([]);
      expect(result.data.length).toBe(0);
      expect(result.bestCropIds.length).toBe(0);
    });

    it('should handle ties in maximum profit', () => {
      const crops = [
        { id: 1, cropName: 'Crop A', farmingCost: '10', expectedYield: '1', marketPrice: '20' }, // Profit 10
        { id: 2, cropName: 'Crop B', farmingCost: '10', expectedYield: '1', marketPrice: '20' }  // Profit 10
      ];

      const result = compareProfits(crops);
      expect(result.bestCropIds).toEqual([1, 2]);
    });
  });
});
