import { lazy } from 'react';

const routeMetrics = {
  loadedRoutes: new Set(),
  loadTimes: {},
};

const preloadRoute = async (loader, pageName) => {
  const start = performance.now();

  try {
    await loader();

    routeMetrics.loadedRoutes.add(pageName);
    routeMetrics.loadTimes[pageName] = Math.round(
      performance.now() - start
    );

    console.info(
      `[ROUTE_PRELOADED] ${pageName}`
    );
  } catch (error) {
    console.error(
      `[ROUTE_PRELOAD_FAILED] ${pageName}`,
      error
    );
  }
};

const lazyPage = (loader, pageName) =>
  lazy(async () => {
    const start = performance.now();

    try {
      const module = await loader();

      const duration = Math.round(
        performance.now() - start
      );

      routeMetrics.loadedRoutes.add(
        pageName
      );

      routeMetrics.loadTimes[
        pageName
      ] = duration;

      console.info(
        `[ROUTE_LOADED] ${pageName} (${duration}ms)`
      );

      return module;
    } catch (error) {
      console.error(
        `[ROUTE_LOAD_FAILED] ${pageName}`,
        error
      );

      throw error;
    }
  });

export const Home = lazyPage(() => import('../Home'), 'Home');
export const Advisor = lazyPage(() => import('../Advisor'), 'Advisor');
export const How = lazyPage(() => import('../How'), 'How');
export const Dashboard = lazyPage(() => import('../Dashboard'), 'Dashboard');
export const CropGuide = lazyPage(() => import('../CropGuide'), 'CropGuide');
export const Schemes = lazyPage(() => import('../GovernmentSchemes'), 'GovernmentSchemes');
export const Resources = lazyPage(() => import('../Resources'), 'Resources');
export const Auth = lazyPage(() => import('../Auth'), 'Auth');
export const ProfileSetup = lazyPage(() => import('../ProfileSetup'), 'ProfileSetup');
export const Calendar = lazyPage(() => import('../FarmingCalendar'), 'FarmingCalendar');
export const Feedback = lazyPage(() => import('../Feedback'), 'Feedback');
export const AdminFeedback = lazyPage(() => import('../AdminFeedback'), 'AdminFeedback');
export const MarketPrices = lazyPage(() => import('../MarketPrices'), 'MarketPrices');
export const FarmingMap = lazyPage(() => import('../FarmingMap'), 'FarmingMap');
export const FarmingNews = lazyPage(() => import('../FarmingNews'), 'FarmingNews');
export const CropProfitCalculator = lazyPage(() => import('../CropProfitCalculator'), 'CropProfitCalculator');
export const Community = lazyPage(() => import('../Community'), 'Community');
export const SoilAnalysis = lazyPage(() => import('../SoilAnalysis'), 'SoilAnalysis');
export const FAQ = lazyPage(() => import('../FAQ'), 'FAQ');
export const Terms = lazyPage(() => import('../Terms'), 'Terms');
export const PrivacyPolicy = lazyPage(() => import('../PrivacyPolicy'), 'PrivacyPolicy');
export const Contributors = lazyPage(() => import('../Contributors'), 'Contributors');
export const QRTraceability = lazyPage(() => import('../QRTraceability'), 'QRTraceability');
export const ContactUs = lazyPage(() => import('../ContactUs'), 'ContactUs');
export const AboutUs = lazyPage(() => import('../AboutUs'), 'AboutUs');
export const SeasonalCropPlanner = lazyPage(() => import('../SeasonalCropPlanner'), 'SeasonalCropPlanner');
export const SoilGuide = lazyPage(() => import('../SoilGuide'), 'SoilGuide');
export const CropDiseaseAwareness = lazyPage(() => import('../CropDiseaseAwareness'), 'CropDiseaseAwareness');
export const PestDetection = lazyPage(() => import('../PestDetection'), 'PestDetection');
export const PestCalendar = lazyPage(() => import('../PestCalendar'), 'PestCalendar');
export const EquipmentManagement = lazyPage(() => import('../EquipmentManagement'), 'EquipmentManagement');
export const Helpline = lazyPage(() => import('../Helpline'), 'Helpline');
export const Glossary = lazyPage(() => import('../Glossary'), 'Glossary');
export const RiskIndex = lazyPage(() => import('../RiskIndex'), 'RiskIndex');
export const CropRotation = lazyPage(() => import('../CropRotation'), 'CropRotation');
export const SeedVerifier = lazyPage(() => import('../SeedVerifier'), 'SeedVerifier');
export const FarmFinance = lazyPage(() => import('../FarmFinance'), 'FarmFinance');
export const YieldPredictor = lazyPage(() => import('../YieldPredictor'), 'YieldPredictor');
export const SmartFarmAutopilot = lazyPage(() => import('../SmartFarmAutopilot'), 'SmartFarmAutopilot');
export const SustainabilityAnalytics = lazyPage(() => import('../SustainabilityAnalytics'), 'SustainabilityAnalytics');
export const Leaderboard = lazyPage(() => import('../Leaderboard'), 'Leaderboard');
export const ReferralHub = lazyPage(() => import('../ReferralHub'), 'ReferralHub');
export const Blog = lazyPage(() => import('../Blog'), 'Blog');
export const BlogDetail = lazyPage(() => import('../BlogDetail'), 'BlogDetail');
export const ProfileSettings = lazyPage(() => import('../ProfileSettings'), 'ProfileSettings');
export const NotFound = lazyPage(() => import('../NotFound'), 'NotFound');
export const RetrainingPipelineMonitor = lazyPage(() => import('../RetrainingPipelineMonitor'), 'RetrainingPipelineMonitor');
export const PredictionExplainer = lazyPage(() => import('../PredictionExplainer'), 'PredictionExplainer');
export const FeatureDriftMonitor = lazyPage(() => import('../FeatureDriftMonitor'), 'FeatureDriftMonitor');
export const CropInsuranceClaim = lazyPage(() => import('../CropInsuranceClaim'), 'CropInsuranceClaim');

export const getRouteMetrics = () => ({
  loadedRoutes: [
    ...routeMetrics.loadedRoutes,
  ],
  loadTimes: {
    ...routeMetrics.loadTimes,
  },
});

export const preloadDashboardRoutes =
  async () => {
    await Promise.all([
      preloadRoute(
        () => import('../Dashboard'),
        'Dashboard'
      ),
      preloadRoute(
        () => import('../Advisor'),
        'Advisor'
      ),
      preloadRoute(
        () => import('../ProfileSettings'),
        'ProfileSettings'
      ),
      preloadRoute(
        () => import('../Leaderboard'),
        'Leaderboard'
      ),
    ]);
  };

export const preloadCommunityRoutes =
  async () => {
    await Promise.all([
      preloadRoute(
        () => import('../Community'),
        'Community'
      ),
      preloadRoute(
        () => import('../Blog'),
        'Blog'
      ),
      preloadRoute(
        () => import('../Contributors'),
        'Contributors'
      ),
    ]);
  };