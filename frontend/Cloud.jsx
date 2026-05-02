import React from "react";

export default function Cloud({ className = "", style = {} }) {
  return (
    <svg
      className={className}
      style={style}
      viewBox="0 0 100 60"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <ellipse cx="35" cy="35" rx="25" ry="15" fill="rgba(255,255,255,0.9)"/>
      <ellipse cx="55" cy="30" rx="30" ry="18" fill="rgba(255,255,255,0.9)"/>
      <ellipse cx="75" cy="38" rx="22" ry="13" fill="rgba(255,255,255,0.9)"/>
    </svg>
  );
}
