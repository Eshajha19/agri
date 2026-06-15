export class TranslationService {
  constructor() {
    this.providers = [
      this.googleTranslateProvider,
      this.firebaseTranslateProvider,
      this.localFallbackProvider,
    ];
  }

  async translate(text, targetLang) {
    for (const provider of this.providers) {
      try {
        const result = await provider(text, targetLang);
        if (result) return result;
      } catch (err) {
        console.warn(`Provider failed: ${provider.name}`, err);
      }
    }
    return text; // final fallback: return original text
  }

  async googleTranslateProvider(text, targetLang) {
    if (window.google && window.google.translate) {
      return window.google.translate(text, targetLang);
    }
    throw new Error("Google Translate unavailable");
  }

  async firebaseTranslateProvider(text, targetLang) {
    try {
      const resp = await fetch(`/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, targetLang }),
      });
      if (!resp.ok) throw new Error("Firebase translation failed");
      const data = await resp.json();
      return data.translatedText;
    } catch {
      throw new Error("Firebase translation unavailable");
    }
  }

  async localFallbackProvider(text, targetLang) {
    // Simple dictionary-based fallback
    const dictionary = {
      hi: { "Hello": "नमस्ते" },
      es: { "Hello": "Hola" },
    };
    return dictionary[targetLang]?.[text] || null;
  }
}
