import React, { useState } from "react";

function ClaimChatbot() {
  const [step, setStep] = useState(0);
  const [formData, setFormData] = useState({
    crop: "",
    damageDate: "",
    photos: [],
    review: false,
  });

  const questions = [
    "Which crop was affected?",
    "When did the damage occur?",
    "Upload supporting photos.",
    "Review claim details."
  ];

  const handleNext = (value) => {
    const keys = ["crop", "damageDate", "photos", "review"];
    setFormData({ ...formData, [keys[step]]: value });
    setStep(step + 1);
  };

  return (
    <div className="chatbot-container">
      <h2>Claim Filing Assistant</h2>
      <div className="chat-window">
        <p>{questions[step]}</p>

        {step === 0 && (
          <input
            type="text"
            placeholder="Enter crop name"
            onBlur={(e) => handleNext(e.target.value)}
          />
        )}

        {step === 1 && (
          <input
            type="date"
            onChange={(e) => handleNext(e.target.value)}
          />
        )}

        {step === 2 && (
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={(e) => handleNext([...e.target.files])}
          />
        )}

        {step === 3 && (
          <div>
            <h3>Review Claim</h3>
            <p>Crop: {formData.crop}</p>
            <p>Damage Date: {formData.damageDate}</p>
            <p>Photos: {formData.photos.length} uploaded</p>
            <button onClick={() => alert("Claim submitted!")}>
              Submit Claim
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default ClaimChatbot;
