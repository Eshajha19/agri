import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

// Import translation files
import enTranslations from '../locales/en/translation.json';
import hiTranslations from '../locales/hi/translation.json';
import mrTranslations from '../locales/mr/translation.json';
import bnTranslations from '../locales/bn/translation.json';
import taTranslations from '../locales/ta/translation.json';
import teTranslations from '../locales/te/translation.json';
import guTranslations from '../locales/gu/translation.json';
import paTranslations from '../locales/pa/translation.json';
import knTranslations from '../locales/kn/translation.json';
import mlTranslations from '../locales/ml/translation.json';
import orTranslations from '../locales/or/translation.json';
import asTranslations from '../locales/as/translation.json';

const getInitialLanguage = () => {
  try {
    const stored = localStorage.getItem('preferredLanguage');
    return stored || 'en';
  } catch {
    return 'en';
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: enTranslations },
      hi: { translation: hiTranslations },
      mr: { translation: mrTranslations },
      bn: { translation: bnTranslations },
      ta: { translation: taTranslations },
      te: { translation: teTranslations },
      gu: { translation: guTranslations },
      pa: { translation: paTranslations },
      kn: { translation: knTranslations },
      ml: { translation: mlTranslations },
      or: { translation: orTranslations },
      as: { translation: asTranslations },
    },
    lng: getInitialLanguage(),
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
  });

export default i18n;
