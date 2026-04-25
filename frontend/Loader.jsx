import React from 'react';
import './Loader.css';

const Loader = ({ fullPage = false, message = "Loading..." }) => {
  return (
    <div className={`loader-container ${fullPage ? 'full-page' : ''}`}>
      <div className="loader-content">
        <div className="spinner">
          <div className="inner-circle"></div>
          <svg className="leaf-svg" viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-8 8Z" />
            <path d="M11 20c-1.2-1.2-2-3-2-5" />
          </svg>
        </div>
        {message && <p className="loader-message">{message}</p>}
      </div>
    </div>
  );
};

export default Loader;
