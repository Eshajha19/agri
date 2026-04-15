import React, { useEffect } from "react";

const GoogleTranslate = ({ lang }) => {
  useEffect(() => {
    if (!document.querySelector("#google-translate-script")) {
      const script = document.createElement("script");
      script.id = "google-translate-script";
      script.src =
        "//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
      script.async = true;
      document.body.appendChild(script);
    }

    window.googleTranslateElementInit = () => {
      new window.google.translate.TranslateElement(
        {
          pageLanguage: "en",
          includedLanguages: "en,hi,mr,bn,ta,te,gu,pa,kn,ml,or",
          autoDisplay: false,
        },
        "google_translate_element"
      );
    };
  }, []);

  // Apply translation whenever lang changes
  useEffect(() => {
    if (lang) {
      const select = document.querySelector(".goog-te-combo");
      if (select) {
        select.value = lang;
        select.dispatchEvent(new Event("change"));
      }
    }
  }, [lang]);

  return (
    <div>
      {/* Hidden Google Translate widget */}
      <div id="google_translate_element" style={{ display: "none" }}></div>
    </div>
  );
};

export default GoogleTranslate;
