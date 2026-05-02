import React, { useState, useRef, useEffect } from "react";
import "./SoilChatbot.css";

function SoilChatbot({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [soilImage, setSoilImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const toBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = (error) => reject(error);
    });

  const callGeminiAPI = async (userText, imageFile) => {
    try {
      const parts = [];

      // Enhanced Prompt
      parts.push({
        text: `You are an agricultural soil expert AI. Analyze and respond clearly with:\n
1. Soil Type\n2. Fertility Status\n3. Suitable Crops\n4. Improvement Tips\n\nUser Query: ${
          userText || "Analyze this soil image"
        }`,
      });

      if (imageFile) {
        parts.push({
          inline_data: {
            data: await toBase64(imageFile),
            mime_type: imageFile.type,
          },
        });
      }

      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parts }),
      });

      if (!response.ok) throw new Error("API failed");

      const data = await response.json();
      return data.reply || "🤖 No response";
    } catch (err) {
      console.error(err);
      return "❌ Error connecting to AI service.";
    }
  };

  const addMessage = (text, from = "bot") => {
    setMessages((prev) => [
      ...prev,
      { text, from, time: new Date().toLocaleTimeString() },
    ]);
  };

  const handleUserInput = async (e) => {
    e.preventDefault();
    const userInput = e.target.userInput.value.trim();

    if (!userInput && !soilImage) {
      addMessage("⚠️ Please enter a query or upload an image.");
      return;
    }

    addMessage(userInput || "[Image sent]", "user");
    e.target.reset();

    setLoading(true);
    const response = await callGeminiAPI(userInput, soilImage);
    setLoading(false);

    addMessage(response, "bot");
    setSoilImage(null);
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
      {/* Header */}
      <div className="chat-header">
        <h2>🌱 Soil Health Chatbot</h2>
        <button className="close-btn" onClick={onClose}>✖</button>
      </div>

      {/* Chat Window */}
      <div className="chat-window">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.from}`}>
            <div>{msg.text}</div>
            <span className="time">{msg.time}</span>
          </div>
        ))}

        {loading && (
          <div className="chat-message bot">🌾 Analyzing soil...</div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Image Preview */}
      {soilImage && (
        <div className="image-preview">
          <img src={URL.createObjectURL(soilImage)} alt="preview" />
        </div>
      )}

      {/* Quick Actions */}
      <div className="quick-actions">
        <button onClick={() => addMessage("Best crops for clay soil?", "user")}>🌱 Crops</button>
        <button onClick={() => addMessage("How to improve soil fertility?", "user")}>🧪 Tips</button>
      </div>

      {/* Input */}
      <form className="chat-input" onSubmit={handleUserInput}>
        <label htmlFor="file-upload" className="file-label">📷</label>
        <input id="file-upload" type="file" accept="image/*" onChange={handleImageUpload} />

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