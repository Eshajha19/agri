import React from "react";

import { useTranslation } from 'react-i18next';
export default function Birds() {
  const { t } = useTranslation();
  return (
    <div className="birds-layer" aria-hidden="true">
      <svg className="bird bird-1" viewBox="0 0 24 24" fill="none">
        <path d="M2 12l20-10" stroke="#333" strokeWidth="2" strokeLinecap="round"/>
      </svg>
      <svg className="bird bird-2" viewBox="0 0 24 24" fill="none">
        <path d="M2 12l20-10" stroke="#333" strokeWidth="2" strokeLinecap="round"/>
      </svg>
      <svg className="bird bird-3" viewBox="0 0 24 24" fill="none">
        <path d="M2 12l20-10" stroke="#333" strokeWidth="2" strokeLinecap="round"/>
      </svg>
    </div>
  );
}
