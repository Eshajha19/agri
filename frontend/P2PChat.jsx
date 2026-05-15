import React, { useState, useEffect, useRef } from "react";
import { Send, Lock, ShieldCheck, X, KeyRound } from "lucide-react";
import { db, auth } from "./lib/firebase";
import { 
  collection, 
  addDoc, 
  query, 
  where, 
  orderBy, 
  onSnapshot, 
  Timestamp,
  doc,
  setDoc,
  getDoc
} from "firebase/firestore";
import { isFirebaseConfigured } from "./lib/firebase";
import { cryptoService } from "./utils/cryptoService";
import "./P2PChat.css";

const P2PChat = ({ recipient, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [isRecipientVerified, setIsRecipientVerified] = useState(true); // Mock verification
  const [sharedKey, setSharedKey] = useState(null);
  const [keyStatus, setKeyStatus] = useState("initializing"); // initializing, ready, waiting
  
  const messagesEndRef = useRef(null);
  const currentUser = auth?.currentUser;

  const hasValidRecipient = recipient && recipient.userId;
  const effectiveRecipient = hasValidRecipient ? recipient : { userId: "default", userName: "Chat" };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // 1. Initialize Cryptographic Keys
  useEffect(() => {
    let isMounted = true;

    const initCrypto = async () => {
      if (!currentUser) {
        if (isMounted) setKeyStatus("auth_required");
        console.warn("P2PChat: No currentUser found.");
        return;
      }
      if (!effectiveRecipient.userId) {
        console.warn("P2PChat: No recipient userId.");
        return;
      }
      
      try {
        if (isMounted) setKeyStatus("generating_keys");
        // A. Load or generate our ECDH key pair
        // Private key is stored as a non-extractable CryptoKey in IndexedDB —
        // its raw bytes are never accessible to JavaScript.
        let privateKey = await cryptoService.loadPrivateKey(currentUser.uid);
        let publicJwk = null;

        if (!privateKey) {
          // Check for a legacy key in localStorage and migrate it
          const legacyPrivateJwk = localStorage.getItem(`ecdh_private_${currentUser.uid}`);
          const legacyPublicJwk  = localStorage.getItem(`ecdh_public_${currentUser.uid}`);

          if (legacyPrivateJwk && legacyPublicJwk) {
            // Re-import as non-extractable and persist to IndexedDB
            privateKey = await cryptoService.importPrivateKey(JSON.parse(legacyPrivateJwk));
            await cryptoService.savePrivateKey(currentUser.uid, privateKey);
            publicJwk = JSON.parse(legacyPublicJwk);
            // Remove plaintext key material from localStorage
            localStorage.removeItem(`ecdh_private_${currentUser.uid}`);
            localStorage.removeItem(`ecdh_public_${currentUser.uid}`);
          } else {
            // Generate a fresh key pair
            const keyPair = await cryptoService.generateECDHKeyPair();
            await cryptoService.savePrivateKey(currentUser.uid, keyPair.privateKey);
            privateKey = keyPair.privateKey;
            publicJwk = await cryptoService.exportKey(keyPair.publicKey);
          }
        } else {
          // Private key already in IndexedDB — retrieve public JWK from Firebase
          if (isFirebaseConfigured()) {
            const pubKeyRef = doc(db, "public_keys", currentUser.uid);
            const snap = await getDoc(pubKeyRef);
            if (snap.exists()) {
              publicJwk = snap.data().jwk;
            } else {
              // Public key missing — regenerate pair
              const keyPair = await cryptoService.generateECDHKeyPair();
              await cryptoService.savePrivateKey(currentUser.uid, keyPair.privateKey);
              privateKey = keyPair.privateKey;
              publicJwk = await cryptoService.exportKey(keyPair.publicKey);
            }
          }
        }

        if (isMounted) setKeyStatus("publishing_key");
        // B. Publish our public key to Firestore for peers to find
        if (isFirebaseConfigured()) {
          if (publicJwk) {
            const pubKeyRef = doc(db, "public_keys", currentUser.uid);
            await setDoc(pubKeyRef, { jwk: publicJwk }, { merge: true });
          }
        } else {
          // Local test fallback — public key only (never the private key)
          if (publicJwk) {
            localStorage.setItem(`remote_ecdh_public_${currentUser.uid}`, JSON.stringify(publicJwk));
          }
        }

        if (isMounted) setKeyStatus("fetching_peer_key");
        // C. Retrieve the recipient's public key
        let recipientPubKeyJwk = null;
        if (effectiveRecipient.userId === "default" || effectiveRecipient.userId === "advisor") {
          // Mock behavior for the 'default' or 'advisor' testing chat:
          // Treat it as a "Note to Self" by using our own public key!
          recipientPubKeyJwk = publicJwk;
        } else if (isFirebaseConfigured()) {
          const recipientRef = doc(db, "public_keys", effectiveRecipient.userId);
          const recipientSnap = await getDoc(recipientRef);
          if (recipientSnap.exists()) {
            recipientPubKeyJwk = recipientSnap.data().jwk;
          }
        } else {
          const localRec = localStorage.getItem(`remote_ecdh_public_${effectiveRecipient.userId}`);
          if (localRec) recipientPubKeyJwk = JSON.parse(localRec);
        }

        // D. Derive the shared symmetric key
        if (recipientPubKeyJwk) {
          const recipientPublicKey = await cryptoService.importPublicKey(recipientPubKeyJwk);
          const derivedKey = await cryptoService.deriveSharedSecret(privateKey, recipientPublicKey);
          if (isMounted) {
            setSharedKey(derivedKey);
            setKeyStatus("ready");
          }
        } else {
          if (isMounted) setKeyStatus("waiting");
          console.warn("Recipient has not published a public key yet.");
        }
      } catch (error) {
        console.error("Crypto init failed:", error);
        if (isMounted) setKeyStatus("error");
      }
    };

    initCrypto();
    return () => { isMounted = false; };
  }, [currentUser, effectiveRecipient.userId]);

  // 2. Load Messages (Only when sharedKey is ready)
  useEffect(() => {
    if (!currentUser || !effectiveRecipient.userId || !sharedKey) return;

    const chatId = [currentUser.uid, effectiveRecipient.userId].sort().join("-");
    let isMounted = true;

    if (isFirebaseConfigured()) {
      const q = query(
        collection(db, "direct_messages"),
        where("chatId", "==", chatId),
        orderBy("createdAt", "asc")
      );

      const unsubscribe = onSnapshot(
        q,
        async (snapshot) => {
          const decryptedDocs = await Promise.all(snapshot.docs.map(async (doc) => {
            const data = doc.data();
            try {
              if (data.encryptedContent && data.encryptedContent.iv) {
                const decryptedText = await cryptoService.decryptMessage(data.encryptedContent, sharedKey);
                return { id: doc.id, ...data, content: decryptedText };
              } else {
                return { id: doc.id, ...data, content: "[Legacy Insecure Format]" };
              }
            } catch (e) {
              return { id: doc.id, ...data, content: "[Decryption Failed]" };
            }
          }));

          if (isMounted) {
            setMessages(decryptedDocs);
            setTimeout(scrollToBottom, 100);
          }
        },
        (error) => {
          // Surface Firestore query errors (e.g. missing composite index in
          // production) so they are visible in logs rather than silently
          // leaving the chat empty.
          console.error(
            "P2PChat: Firestore query failed for chatId=%s — %s\n" +
            "If this is a 'requires an index' error, ensure firestore.indexes.json " +
            "has been deployed via: firebase deploy --only firestore:indexes",
            chatId,
            error.message
          );
        }
      );

      return () => {
        isMounted = false;
        unsubscribe();
      };
    } else {
      // Local Fallback Mode for Testing
      const loadLocalMessages = async () => {
        const localData = localStorage.getItem(`p2p_chat_${chatId}`);
        if (localData) {
          const parsedData = JSON.parse(localData);
          const decryptedMessages = await Promise.all(parsedData.map(async (msg) => {
            try {
              if (msg.encryptedContent && msg.encryptedContent.iv) {
                const decryptedText = await cryptoService.decryptMessage(msg.encryptedContent, sharedKey);
                return { ...msg, content: decryptedText };
              } else {
                return { ...msg, content: "[Legacy Format]" };
              }
            } catch (e) {
              return { ...msg, content: "[Decryption Failed]" };
            }
          }));
          if (isMounted) {
            setMessages(decryptedMessages);
            setTimeout(scrollToBottom, 100);
          }
        }
      };

      loadLocalMessages();
      const interval = setInterval(loadLocalMessages, 2000);
      return () => {
        isMounted = false;
        clearInterval(interval);
      };
    }
  }, [sharedKey, currentUser, effectiveRecipient.userId]);

  // 3. Send Message
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!newMessage.trim() || !currentUser || !sharedKey) return;

    const chatId = [currentUser.uid, effectiveRecipient.userId].sort().join("-");
    const messageText = newMessage;
    setNewMessage(""); // Clear early for UX

    try {
      const encrypted = await cryptoService.encryptMessage(messageText, sharedKey);

      if (isFirebaseConfigured()) {
        await addDoc(collection(db, "direct_messages"), {
          chatId,
          senderId: currentUser.uid,
          senderName: currentUser.displayName || currentUser.email?.split('@')[0] || "User",
          recipientId: effectiveRecipient.userId,
          encryptedContent: encrypted, // Stores { ciphertext, iv }
          createdAt: Timestamp.now()
        });
      } else {
        // Local Mode
        const localKey = `p2p_chat_${chatId}`;
        const existing = JSON.parse(localStorage.getItem(localKey) || "[]");
        const newMsg = {
          id: Date.now(),
          senderId: currentUser.uid,
          senderName: "You",
          encryptedContent: encrypted,
          createdAt: { toDate: () => new Date() }
        };
        localStorage.setItem(localKey, JSON.stringify([...existing, newMsg]));
        setMessages(prev => [...prev, { ...newMsg, content: messageText }]);
        setTimeout(scrollToBottom, 100);
      }
    } catch (err) {
      console.error("Encryption/Send failed:", err);
      // Revert input on failure
      setNewMessage(messageText);
    }
  };

  return (
    <div className="p2p-chat-container">
      <div className="p2p-chat-header">
        <div className="recipient-info">
          <div className="user-avatar">
            {effectiveRecipient.userName ? effectiveRecipient.userName[0].toUpperCase() : "U"}
            {isRecipientVerified && <ShieldCheck className="verified-badge" size={14} />}
          </div>
          <div>
            <h3>{effectiveRecipient.userName}</h3>
            <span className="security-status">
              {keyStatus === "ready" ? (
                <><Lock size={12} /> True End-to-End Encrypted</>
              ) : keyStatus === "waiting" ? (
                <><KeyRound size={12} /> Waiting for peer key...</>
              ) : (
                <><KeyRound size={12} /> Securing connection...</>
              )}
            </span>
          </div>
        </div>
        <button className="close-chat-btn" onClick={onClose}><X size={20} /></button>
      </div>

      <div className="p2p-chat-messages">
        <div className="encryption-notice">
          <Lock size={16} />
          <p>Messages are end-to-end encrypted using ECDH and AES-GCM. No one outside of this chat can read them.</p>
        </div>
        
        {messages.map((msg) => (
          <div 
            key={msg.id} 
            className={`message-bubble ${msg.senderId === currentUser?.uid ? 'sent' : 'received'}`}
          >
            <div className="message-content">
              <p>{msg.content}</p>
              <span className="message-time">
                {msg.createdAt?.toDate ? msg.createdAt.toDate().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : "Recently"}
              </span>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form className="p2p-chat-input" onSubmit={handleSendMessage}>
        <input 
          type="text" 
          placeholder={
            keyStatus === "auth_required" ? "Please log in to chat securely." :
            keyStatus === "ready" ? "Type a secure message..." : 
            `Securing: ${keyStatus.replace("_", " ")}...`
          } 
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          disabled={!sharedKey}
          required
        />
        <button type="submit" className="send-msg-btn" aria-label="Send Message" disabled={!sharedKey}>
          <Send className="send-icon-svg" size={40} />
        </button>
      </form>
    </div>
  );
};

export default P2PChat;

