import React, { useRef, useEffect } from "react";
import "./SoilChatbot.css";
import { FaMicrophone, FaStop, FaVolumeUp, FaPaperPlane, FaImage } from "react-icons/fa";
import { useChatbot } from "./hooks/useChatbot";

function SoilChatbot({ onClose }) {
  const {
    messages,
    userInput,
    setUserInput,
    soilImage,
    setSoilImage,
    isListening,
    toggleListening,
    isSpeaking,
    stopSpeaking,
    isLoading,
    handleSendMessage,
  } = useChatbot();

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (textOverride) => {
    const textToSend = textOverride || userInput;
    if (!textToSend && !soilImage) return;

    await handleSendMessage(textToSend, soilImage);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSoilImage(file);
    }
  };

  const suggestions = [
    "🌤️ Weather-based farming advice",
    "🌾 Recommended crops for this month",
    "🧪 How to improve my soil health?",
    "🐛 Pest control for my crops"
  ];

  return (
    <div className="soil-chatbot">
      <div className="chat-header">
        <div className="header-info">
          <h2>🌱 Agri Assistant <FaVolumeUp style={{ fontSize: '0.9rem', marginLeft: '8px', opacity: 0.8 }} /></h2>
          <span className="status">AI Agricultural Expert</span>
        </div>
        <button className="close-btn" onClick={onClose}>✖</button>
      </div>

      <div className="chat-window">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.from}`}>
            <div className="message-content">{msg.text}</div>
          </div>
        ))}
        {isLoading && <div className="chat-message bot loading-dots">Thinking...</div>}
        {isListening && <div className="chat-message user listening">Listening... 🎤</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="suggestions-bar">
        {suggestions.map((s, i) => (
          <button key={i} className="suggestion-chip" onClick={() => handleSend(s)}>
            {s}
          </button>
        ))}
      </div>

      <div className="chat-controls">
        <div className="voice-controls">
          <button
            className={`control-btn mic-btn ${isListening ? "active" : ""}`}
            onClick={toggleListening}
            title="Start / Stop Voice Input"
          >
            <FaMicrophone />
          </button>

          {isSpeaking && (
            <button className="control-btn stop-btn" onClick={stopSpeaking} title="Stop Speaking">
              <FaStop />
            </button>
          )}
        </div>

        <div className="input-area">
          <label htmlFor="file-upload" className="image-btn" title="Upload Soil/Crop Image">
            <FaImage />
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleImageUpload}
          />

          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Ask about crops, weather, soil..."
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
          />

          <button className="send-btn" onClick={() => handleSend()}>
            <FaPaperPlane />
          </button>
        </div>
      </div>
    </div>
  );
}

export default SoilChatbot;

