import React from "react";
import "./ErrorToast.css";

export default function ErrorToast({ message, onClose }) {
  if (!message) return null;

  return (
    <div className="error-toast">
      <span>{message}</span>
      <button onClick={onClose} aria-label="Close error">×</button>
    </div>
  );
}
