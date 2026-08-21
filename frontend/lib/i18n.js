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
import { SUPPORTED_LANGUAGE_CODES } from './languageOptions';

const SUPPORTED_LANGUAGES = SUPPORTED_LANGUAGE_CODES;

const getInitialLanguage = () => {
  try {
    const storedLanguage = localStorage.getItem('agri:preferredLanguage');
    if (SUPPORTED_LANGUAGES.includes(storedLanguage)) {
      return storedLanguage;
    }
  } catch {
    // Storage may be unavailable in privacy mode or during SSR.
  }

  const browserLanguage = typeof navigator !== 'undefined'
    ? navigator.language?.split('-')[0]
    : null;

  return SUPPORTED_LANGUAGES.includes(browserLanguage) ? browserLanguage : 'en';
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
    supportedLngs: SUPPORTED_LANGUAGES,
    load: 'languageOnly',
    interpolation: {
      escapeValue: false,
    },
  });

i18n.on('languageChanged', (language) => {
  const normalizedLanguage = language.split('-')[0];
  document.documentElement.lang = normalizedLanguage;
  try {
    localStorage.setItem('agri:preferredLanguage', normalizedLanguage);
  } catch {
    // Language changes still work when storage is unavailable.
  }
});

document.documentElement.lang = i18n.language;

export default i18n;
