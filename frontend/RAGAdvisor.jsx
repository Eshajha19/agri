import React, { useState, useRef, useEffect } from "react";
import { Send, BookOpen, X, Loader, ExternalLink, ChevronDown } from "lucide-react";
import "./RAGAdvisor.css";

const EXAMPLE_QUERIES = [
  "How do I control rice blast disease?",
  "Best fertilizer for wheat crop?",
  "How to save water in cotton irrigation?",
  "What government schemes are available for farmers?",
  "How to fix zinc deficiency in soil?",
  "Best practices for soybean intercropping?",
];

const TOPIC_COLORS = {
  fertilizer: "#10b981",
  pest_disease: "#ef4444",
  irrigation: "#3b82f6",
  government_scheme: "#8b5cf6",
  soil_health: "#f59e0b",
  crop_management: "#06b6d4",
  crop_variety: "#ec4899",
  climate_adaptation: "#6366f1",
  cropping_system: "#84cc16",
  post_harvest: "#f97316",
};

function CitationCard({ citation }) {
  const color = TOPIC_COLORS[citation.topic] || "#64748b";
  return (
    <div className="citation-card" style={{ borderLeftColor: color }}>
      <div className="citation-index" style={{ background: color }}>[{citation.index}]</div>
      <div className="citation-body">
        <p className="citation-title">{citation.title}</p>
        <p className="citation-source">{citation.citation}</p>
        <p className="citation-meta">{citation.source} · {citation.year}</p>
      </div>
    </div>
  );
}

function MessageBubble({ msg }) {
  if (msg.role === "user") {
    return <div className="msg-bubble user">{msg.text}</div>;
  }
  return (
    <div className="msg-bubble assistant">
      <div className="msg-answer">{msg.text}</div>
      {msg.citations?.length > 0 && (
        <div className="citations-section">
          <p className="citations-label">📚 Research Sources</p>
          {msg.citations.map((c) => <CitationCard key={c.index} citation={c} />)}
        </div>
      )}
    </div>
  );
}

const RAGAdvisor = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hello! I am your AI Agricultural Advisor powered by verified research sources. Ask me anything about crop management, fertilizers, pest control, or government schemes — I'll answer with citations!",
      citations: [],
    },
  ]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (!isOpen) return null;

  const handleSend = async (q) => {
    const text = (q || query).trim();
    if (!text) return;
    setQuery("");
    setError("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);
    try {
      const res = await fetch("/api/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, top_k: 3 }),
      });
      if (!res.ok) throw new Error("Backend unavailable");
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, citations: data.citations },
      ]);
    } catch {
      // Fallback: client-side answer using local knowledge
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "⚠️ The research backend is currently offline. Please ensure the Fasal Saathi API server is running on port 8000 to get citation-backed responses.",
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rag-overlay">
      <div className="rag-modal">
        {/* Header */}
        <div className="rag-header">
          <div className="rag-header-left">
            <BookOpen size={22} />
            <div>
              <h2>AI Research Advisor</h2>
              <p>Responses backed by verified agricultural research</p>
            </div>
          </div>
          <button className="rag-close" onClick={onClose}><X /></button>
        </div>

        {/* Example queries */}
        <div className="rag-examples">
          {EXAMPLE_QUERIES.slice(0, 3).map((eq) => (
            <button key={eq} className="example-chip" onClick={() => handleSend(eq)}>
              {eq}
            </button>
          ))}
        </div>

        {/* Messages */}
        <div className="rag-messages">
          {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
          {loading && (
            <div className="msg-bubble assistant loading-bubble">
              <Loader size={18} className="spin" />
              <span>Searching research database...</span>
            </div>
          )}
          {error && <p className="rag-error">{error}</p>}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="rag-input-bar">
          <input
            type="text"
            placeholder="Ask about crop management, fertilizers, pests, schemes..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={() => handleSend()}
            disabled={loading || !query.trim()}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default RAGAdvisor;
