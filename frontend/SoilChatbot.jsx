import React, { useState, useEffect, useRef } from "react";
import "./SoilChatbot.css";
import { FaMicrophone, FaStop, FaVolumeUp, FaPaperPlane, FaImage } from "react-icons/fa";

function SoilChatbot({ onClose }) {
  const [messages, setMessages] = useState([
    { text: "Hello! I'm your Agricultural Assistant. How can I help you today?", from: "bot" }
  ]);
  const [userInput, setUserInput] = useState("");
  const [soilImage, setSoilImage] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const recognitionRef = useRef(null);
  const synthRef = useRef(window.speechSynthesis);
  const utteranceRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = "en-US";

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setUserInput(transcript);
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, []);

  const toBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = (error) => reject(error);
    });

  const callGeminiAPI = async (userText, imageFile) => {
    setIsLoading(true);
    try {
      const parts = [];

      // Add system prompt context
      const systemPrompt = "You are an AI Agricultural Expert. Provide helpful suggestions for crops, weather-based farming advice, and soil health. Keep answers concise and practical for farmers. Mention climate conditions specifically if relevant.";

      parts.push({ text: `${systemPrompt}\n\nUser Question: ${userText}` });

      if (imageFile) {
        parts.push({
          inline_data: {
            data: await toBase64(imageFile),
            mime_type: imageFile.type,
          },
        });
      }

      const API_KEY = import.meta.env.VITE_API_KEY;

      if (!API_KEY) {
        setIsLoading(false);
        return "⚠️ API Key is missing. Please add your Gemini API key in SoilChatbot.jsx to enable the AI features.";
      }

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=${API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts }],
          }),
        }
      );

      const data = await response.json();
      setIsLoading(false);

      if (data.error) {
        return `❌ API Error: ${data.error.message}`;
      }

      const botText = data?.candidates?.[0]?.content?.parts?.[0]?.text || "🤖 I'm sorry, I couldn't process that.";
      return botText;
    } catch (err) {
      console.error("Gemini API Error:", err);
      setIsLoading(false);
      return "❌ Error: Unable to connect to Gemini API.";
    }
  };

  const addMessage = (text, from = "bot") => {
    setMessages((prev) => [...prev, { text, from }]);
    if (from === "bot") {
      speakText(text);
    }
  };

  const handleSend = async (textOverride) => {
    const textToSend = textOverride || userInput;
    if (!textToSend && !soilImage) return;

    addMessage(textToSend || "[Image sent]", "user");
    setUserInput("");

    const response = await callGeminiAPI(textToSend, soilImage);
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

  const startListening = () => {
    if (recognitionRef.current) {
      setIsListening(true);
      recognitionRef.current.start();
    } else {
      alert("Speech recognition is not supported in this browser.");
    }
  };

  const speakText = (text) => {
    if (synthRef.current) {
      synthRef.current.cancel();

      // Clean text of markdown characters so it doesn't say "asterisk" or "star"
      const cleanText = text
        .replace(/\*/g, "")
        .replace(/#/g, "")
        .replace(/_/g, "")
        .replace(/[`]/g, "");

      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utteranceRef.current = utterance;
      synthRef.current.speak(utterance);
    }
  };

  const stopSpeaking = () => {
    if (synthRef.current) {
      synthRef.current.cancel();
      setIsSpeaking(false);
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
            onClick={isListening ? () => recognitionRef.current.stop() : startListening}
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

