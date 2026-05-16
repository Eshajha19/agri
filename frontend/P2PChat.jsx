import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  Send,
  Lock,
  ShieldCheck,
  X,
  KeyRound,
  AlertTriangle,
} from "lucide-react";

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
  getDoc,
} from "firebase/firestore";

import { isFirebaseConfigured } from "./lib/firebase";
import { cryptoService } from "./utils/cryptoService";

import "./P2PChat.css";

const P2PChat = ({ recipient, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [sharedKey, setSharedKey] = useState(null);

  const [keyStatus, setKeyStatus] = useState(
    "initializing"
  );

  const [loadingMessages, setLoadingMessages] =
    useState(true);

  const messagesEndRef = useRef(null);

  const currentUser = auth?.currentUser;

  const effectiveRecipient = useMemo(() => {
    return recipient?.userId
      ? recipient
      : {
          userId: "default",
          userName: "Secure Chat",
        };
  }, [recipient]);

  const chatId = useMemo(() => {
    if (!currentUser?.uid) return null;

    return [
      currentUser.uid,
      effectiveRecipient.userId,
    ]
      .sort()
      .join("-");
  }, [currentUser, effectiveRecipient.userId]);

  /* =========================================================
      AUTO SCROLL
  ========================================================= */

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  /* =========================================================
      INIT CRYPTO
  ========================================================= */

  useEffect(() => {
    let mounted = true;

    const initCrypto = async () => {
      try {
        if (!currentUser) {
          if (mounted)
            setKeyStatus("auth_required");
          return;
        }

        setKeyStatus("generating_keys");

        let privateKey =
          await cryptoService.loadPrivateKey(
            currentUser.uid
          );

        let publicJwk = null;

        /* ==============================
            GENERATE KEY PAIR
        ============================== */

        if (!privateKey) {
          const keyPair =
            await cryptoService.generateECDHKeyPair();

          privateKey = keyPair.privateKey;

          await cryptoService.savePrivateKey(
            currentUser.uid,
            privateKey
          );

          publicJwk =
            await cryptoService.exportKey(
              keyPair.publicKey
            );
        } else {
          if (isFirebaseConfigured()) {
            const pubRef = doc(
              db,
              "public_keys",
              currentUser.uid
            );

            const snap = await getDoc(pubRef);

            if (snap.exists()) {
              publicJwk = snap.data().jwk;
            } else {
              const keyPair =
                await cryptoService.generateECDHKeyPair();

              privateKey = keyPair.privateKey;

              await cryptoService.savePrivateKey(
                currentUser.uid,
                privateKey
              );

              publicJwk =
                await cryptoService.exportKey(
                  keyPair.publicKey
                );
            }
          }
        }

        /* ==============================
            PUBLISH PUBLIC KEY
        ============================== */

        setKeyStatus("publishing_key");

        if (isFirebaseConfigured() && publicJwk) {
          await setDoc(
            doc(db, "public_keys", currentUser.uid),
            {
              jwk: publicJwk,
            },
            { merge: true }
          );
        }

        /* ==============================
            FETCH RECIPIENT KEY
        ============================== */

        setKeyStatus("fetching_peer_key");

        let recipientPubKeyJwk = null;

        if (
          effectiveRecipient.userId === "default"
        ) {
          recipientPubKeyJwk = publicJwk;
        } else {
          const recipientRef = doc(
            db,
            "public_keys",
            effectiveRecipient.userId
          );

          const recipientSnap = await getDoc(
            recipientRef
          );

          if (recipientSnap.exists()) {
            recipientPubKeyJwk =
              recipientSnap.data().jwk;
          }
        }

        /* ==============================
            DERIVE SHARED SECRET
        ============================== */

        if (!recipientPubKeyJwk) {
          if (mounted)
            setKeyStatus("waiting");

          return;
        }

        const recipientPublicKey =
          await cryptoService.importPublicKey(
            recipientPubKeyJwk
          );

        const derivedKey =
          await cryptoService.deriveSharedSecret(
            privateKey,
            recipientPublicKey
          );

        if (mounted) {
          setSharedKey(derivedKey);
          setKeyStatus("ready");
        }
      } catch (err) {
        console.error(err);

        if (mounted) {
          setKeyStatus("error");
        }
      }
    };

    initCrypto();

    return () => {
      mounted = false;
    };
  }, [currentUser, effectiveRecipient]);

  /* =========================================================
      LOAD MESSAGES
  ========================================================= */

  useEffect(() => {
    if (
      !currentUser ||
      !chatId ||
      !sharedKey
    )
      return;

    setLoadingMessages(true);

    const q = query(
      collection(db, "direct_messages"),
      where("chatId", "==", chatId),
      orderBy("createdAt", "asc")
    );

    const unsubscribe = onSnapshot(
      q,
      async (snapshot) => {
        const decryptedMessages =
          await Promise.all(
            snapshot.docs.map(async (docItem) => {
              const data = docItem.data();

              try {
                const decrypted =
                  await cryptoService.decryptMessage(
                    data.encryptedContent,
                    sharedKey
                  );

                return {
                  id: docItem.id,
                  ...data,
                  content: decrypted,
                };
              } catch {
                return {
                  id: docItem.id,
                  ...data,
                  content:
                    "[Unable to decrypt message]",
                };
              }
            })
          );

        setMessages(decryptedMessages);
        setLoadingMessages(false);
      }
    );

    return () => unsubscribe();
  }, [chatId, sharedKey, currentUser]);

  /* =========================================================
      SEND MESSAGE
  ========================================================= */

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (
      !newMessage.trim() ||
      !sharedKey ||
      !currentUser
    )
      return;

    const messageText = newMessage.trim();

    setNewMessage("");

    try {
      const encrypted =
        await cryptoService.encryptMessage(
          messageText,
          sharedKey
        );

      await addDoc(
        collection(db, "direct_messages"),
        {
          chatId,
          senderId: currentUser.uid,
          senderName:
            currentUser.displayName ||
            currentUser.email?.split("@")[0] ||
            "User",

          recipientId:
            effectiveRecipient.userId,

          encryptedContent: encrypted,

          createdAt: Timestamp.now(),
        }
      );
    } catch (err) {
      console.error(err);

      setNewMessage(messageText);
    }
  };

  /* =========================================================
      UI
  ========================================================= */

  return (
    <div className="p2p-chat-container">
      {/* HEADER */}

      <div className="p2p-chat-header">
        <div className="recipient-info">
          <div className="user-avatar">
            {effectiveRecipient.userName?.[0] ||
              "U"}

            <ShieldCheck
              className="verified-badge"
              size={14}
            />
          </div>

          <div>
            <h3>
              {effectiveRecipient.userName}
            </h3>

            <span className="security-status">
              {keyStatus === "ready" ? (
                <>
                  <Lock size={13} />
                  End-to-End Encrypted
                </>
              ) : keyStatus === "waiting" ? (
                <>
                  <KeyRound size={13} />
                  Waiting for peer key...
                </>
              ) : keyStatus === "error" ? (
                <>
                  <AlertTriangle size={13} />
                  Encryption Error
                </>
              ) : (
                <>
                  <KeyRound size={13} />
                  Securing connection...
                </>
              )}
            </span>
          </div>
        </div>

        <button
          className="close-chat-btn"
          onClick={onClose}
        >
          <X size={20} />
        </button>
      </div>

      {/* MESSAGES */}

      <div className="p2p-chat-messages">
        <div className="encryption-notice">
          <Lock size={16} />

          <p>
            Messages are protected with ECDH +
            AES-GCM encryption.
          </p>
        </div>

        {loadingMessages ? (
          <div className="chat-loader">
            Loading secure messages...
          </div>
        ) : messages.length === 0 ? (
          <div className="empty-chat">
            Start your secure conversation.
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`message-row ${
                msg.senderId === currentUser?.uid
                  ? "sent"
                  : "received"
              }`}
            >
              <div className="message-bubble">
                <p>{msg.content}</p>

                <span className="message-time">
                  {msg.createdAt?.toDate
                    ? msg.createdAt
                        .toDate()
                        .toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                    : ""}
                </span>
              </div>
            </div>
          ))
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* INPUT */}

      <form
        className="p2p-chat-input"
        onSubmit={handleSendMessage}
      >
        <input
          type="text"
          placeholder={
            keyStatus === "auth_required"
              ? "Please login first..."
              : keyStatus === "ready"
              ? "Type secure message..."
              : "Securing chat..."
          }
          value={newMessage}
          onChange={(e) =>
            setNewMessage(e.target.value)
          }
          disabled={keyStatus !== "ready"}
          required
        />

        <button
          type="submit"
          className="send-msg-btn"
          disabled={
            !newMessage.trim() ||
            keyStatus !== "ready"
          }
        >
          <Send size={20} />
        </button>
      </form>
    </div>
  );
};

export default P2PChat;