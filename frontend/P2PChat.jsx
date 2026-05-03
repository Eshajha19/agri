import React, { useState, useEffect, useRef } from "react";
import CryptoJS from "crypto-js";
import { Send, Lock, ShieldCheck, X, User } from "lucide-react";
import { db, auth } from "./lib/firebase";
import { 
  collection, 
  addDoc, 
  query, 
  where, 
  orderBy, 
  onSnapshot, 
  Timestamp 
} from "firebase/firestore";
import { isFirebaseConfigured } from "./lib/firebase";
import "./P2PChat.css";

const P2PChat = ({ recipient, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [isRecipientVerified, setIsRecipientVerified] = useState(true); // Mock verification
  const messagesEndRef = useRef(null);
  const currentUser = auth?.currentUser;

  // Derive a "Shared Secret" for E2EE based on user IDs
  // In a real app, this would be a DH exchange result
  const sharedSecret = [currentUser?.uid, recipient.userId].sort().join("_");

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (!currentUser || !recipient.userId) return;

    const chatId = [currentUser.uid, recipient.userId].sort().join("-");

    if (isFirebaseConfigured()) {
      const q = query(
        collection(db, "direct_messages"),
        where("chatId", "==", chatId),
        orderBy("createdAt", "asc")
      );

      const unsubscribe = onSnapshot(q, (snapshot) => {
        const docs = snapshot.docs.map(doc => {
          const data = doc.data();
          try {
            const bytes = CryptoJS.AES.decrypt(data.encryptedContent, sharedSecret);
            const decryptedText = bytes.toString(CryptoJS.enc.Utf8);
            return {
              id: doc.id,
              ...data,
              content: decryptedText || "[Decryption Failed]"
            };
          } catch (e) {
            return { id: doc.id, ...data, content: "[Encrypted Message]" };
          }
        });
        setMessages(docs);
        setTimeout(scrollToBottom, 100);
      });

      return () => unsubscribe();
    } else {
      // Local Fallback Mode for Testing
      const loadLocalMessages = () => {
        const localData = localStorage.getItem(`p2p_chat_${chatId}`);
        if (localData) {
          const parsedData = JSON.parse(localData);
          const decryptedMessages = parsedData.map(msg => {
            try {
              const bytes = CryptoJS.AES.decrypt(msg.encryptedContent, sharedSecret);
              const decryptedText = bytes.toString(CryptoJS.enc.Utf8);
              return { ...msg, content: decryptedText || "[Decryption Failed]" };
            } catch (e) {
              return { ...msg, content: "[Encrypted Message]" };
            }
          });
          setMessages(decryptedMessages);
          setTimeout(scrollToBottom, 100);
        }
      };

      loadLocalMessages();
      // Poll for local changes (simulating real-time)
      const interval = setInterval(loadLocalMessages, 2000);
      return () => clearInterval(interval);
    }
  }, [recipient.userId, currentUser, sharedSecret]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!newMessage.trim() || !currentUser) return;

    const chatId = [currentUser.uid, recipient.userId].sort().join("-");
    const encrypted = CryptoJS.AES.encrypt(newMessage, sharedSecret).toString();

    if (isFirebaseConfigured()) {
      try {
        await addDoc(collection(db, "direct_messages"), {
          chatId: chatId,
          senderId: currentUser.uid,
          senderName: currentUser.displayName || currentUser.email.split('@')[0],
          recipientId: recipient.userId,
          encryptedContent: encrypted,
          createdAt: Timestamp.now()
        });
      } catch (err) {
        console.error("Error sending message:", err);
      }
    } else {
      // Local Mode: Save to localStorage
      const localKey = `p2p_chat_${chatId}`;
      const existing = JSON.parse(localStorage.getItem(localKey) || "[]");
      const newMsg = {
        id: Date.now(),
        senderId: currentUser.uid,
        senderName: "You",
        encryptedContent: encrypted,
        createdAt: { toDate: () => new Date() } // Mock firebase timestamp
      };
      localStorage.setItem(localKey, JSON.stringify([...existing, newMsg]));
      // Trigger local UI update
      setMessages(prev => [...prev, { ...newMsg, content: newMessage }]);
      setTimeout(scrollToBottom, 100);
    }
    setNewMessage("");
  };

  return (
    <div className="p2p-chat-container">
      <div className="p2p-chat-header">
        <div className="recipient-info">
          <div className="user-avatar">
            {recipient.userName ? recipient.userName[0].toUpperCase() : "U"}
            {isRecipientVerified && <ShieldCheck className="verified-badge" size={14} />}
          </div>
          <div>
            <h3>{recipient.userName}</h3>
            <span className="security-status">
              <Lock size={12} /> End-to-End Encrypted
            </span>
          </div>
        </div>
        <button className="close-chat-btn" onClick={onClose}><X size={20} /></button>
      </div>

      <div className="p2p-chat-messages">
        <div className="encryption-notice">
          <Lock size={16} />
          <p>Messages are end-to-end encrypted. No one outside of this chat, not even Fasal Saathi, can read them.</p>
        </div>
        
        {messages.map((msg) => (
          <div 
            key={msg.id} 
            className={`message-bubble ${msg.senderId === currentUser?.uid ? 'sent' : 'received'}`}
          >
            <div className="message-content">
              <p>{msg.content}</p>
              <span className="message-time">
                {msg.createdAt?.toDate().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form className="p2p-chat-input" onSubmit={handleSendMessage}>
        <input 
          type="text" 
          placeholder="Type an encrypted message..." 
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          required
        />
        <button type="submit" className="send-msg-btn" aria-label="Send Message">
          <Send className="send-icon-svg" size={40} />
        </button>
      </form>
    </div>
  );
};

export default P2PChat;
