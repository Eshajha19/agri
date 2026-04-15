import React, { useState } from "react";
import "../css/SoilChatbot.css";

function SoilChatbot({ onClose }) {   // ⬅️ take onClose as a prop
  const [messages, setMessages] = useState([]);
  const [soilImage, setSoilImage] = useState(null);

  // 🔹 Convert image to base64 so it can be sent to Gemini
  const toBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(",")[1]); // only base64 part
      reader.onerror = (error) => reject(error);
    });

  // 🔹 Real Gemini API call (frontend only)
// 🔹 Real Gemini API call (frontend only - not secure for production)
const callGeminiAPI = async (userText, imageFile) => {
  try {
    const parts = [];

    if (userText) {
      parts.push({ text: userText });
    }

    if (imageFile) {
      parts.push({
        inline_data: {
          data: await toBase64(imageFile),
          mime_type: imageFile.type,
        },
      });
    }

    if (parts.length === 0) return "❌ Please provide text or image.";

    const API_KEY = ""; // put actual Gemini API key here
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key=${API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [
            {
              parts,
            },
          ],
        }),
      }
    );

    const data = await response.json();
    console.log("Gemini response:", data);

    return (
      data?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "🤖 No response from Gemini."
    );
  } catch (err) {
    console.error("Gemini API Error:", err);
    return "❌ Error: Unable to connect to Gemini API.";
  }
};

  const addMessage = (text, from = "bot") => {
    setMessages((prev) => [...prev, { text, from }]);
  };

  const handleUserInput = async (e) => {
    e.preventDefault();
    const userInput = e.target.userInput.value.trim();
    if (!userInput && !soilImage) return;

    addMessage(userInput || "[Image sent]", "user");
    e.target.reset();

    const response = await callGeminiAPI(userInput, soilImage);
    addMessage(response, "bot");
    setSoilImage(null); // clear uploaded image after sending
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSoilImage(file);
      addMessage(`🖼️ Image uploaded: ${file.name}`, "user");
    }
  };

  return (
    <div className="soil-chatbot">
      {/* 🌿 Header with close button */}
      <div className="chat-header">
        <h2>Soil Health Chatbot</h2>
        <button className="close-btn" onClick={onClose}>
          ✖
        </button>
      </div>

      {/* Chat Window */}
      <div className="chat-window">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.from}`}>
            {msg.text}
          </div>
        ))}
      </div>

      {/* Input Section */}
      <form className="chat-input" onSubmit={handleUserInput}>
        <label htmlFor="file-upload" className="file-label">
          📷 Upload
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
        />
        <input
          type="text"
          name="userInput"
          placeholder="Ask about soil or crops..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}

export default SoilChatbot;

