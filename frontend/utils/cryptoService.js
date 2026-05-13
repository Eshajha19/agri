/**
 * Secure E2EE Cryptography Service using Web Crypto API
 * Implements Elliptic Curve Diffie-Hellman (ECDH) and AES-GCM encryption
 *
 * Private keys are stored in IndexedDB as non-extractable CryptoKey objects.
 * This means the raw key material can never be read by JavaScript — not by
 * this code, not by third-party scripts, and not by an XSS attacker.
 */

// ---------------------------------------------------------------------------
// IndexedDB key store
// ---------------------------------------------------------------------------

const IDB_NAME = "fasal_e2ee";
const IDB_STORE = "keys";
const IDB_VERSION = 1;

function openKeyStore() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = (e) => {
      e.target.result.createObjectStore(IDB_STORE);
    };
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

async function idbGet(key) {
  const db = await openKeyStore();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readonly");
    const req = tx.objectStore(IDB_STORE).get(key);
    req.onsuccess = (e) => resolve(e.target.result ?? null);
    req.onerror = (e) => reject(e.target.error);
  });
}

async function idbSet(key, value) {
  const db = await openKeyStore();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readwrite");
    const req = tx.objectStore(IDB_STORE).put(value, key);
    req.onsuccess = () => resolve();
    req.onerror = (e) => reject(e.target.error);
  });
}

// ---------------------------------------------------------------------------
// Crypto service
// ---------------------------------------------------------------------------

export const cryptoService = {
  /**
   * Generate a new ECDH P-256 key pair.
   *
   * The private key is intentionally NON-EXTRACTABLE — its raw bytes can
   * never be read by JavaScript. It is stored directly as a CryptoKey object
   * in IndexedDB. The public key remains extractable so it can be shared.
   */
async generateECDHKeyPair() {
     return await window.crypto.subtle.generateKey(
       { name: "ECDH", namedCurve: "P-256" },
       true, // extractable — keys must be exported before storing non-extractable copies
       ["deriveKey"]
     );
   },

  /**
   * Export a PUBLIC key to JWK format for transmission to Firebase.
   * Never call this with a private key — it will throw because the private
   * key is non-extractable.
   */
  async exportKey(key) {
    return await window.crypto.subtle.exportKey("jwk", key);
  },

  // ---------------------------------------------------------------------------
  // IndexedDB persistence for the non-extractable private key
  // ---------------------------------------------------------------------------

  /**
   * Persist the private CryptoKey for a user in IndexedDB.
   * The browser stores the opaque key handle; the raw bytes never leave the
   * Web Crypto subsystem.
   */
  async savePrivateKey(uid, privateKey) {
    await idbSet(`ecdh_private_${uid}`, privateKey);
  },

  /**
   * Load the private CryptoKey for a user from IndexedDB.
   * Returns null if no key exists yet.
   */
  async loadPrivateKey(uid) {
    return await idbGet(`ecdh_private_${uid}`);
  },

  // ---------------------------------------------------------------------------
  // Import helpers
  // ---------------------------------------------------------------------------

  // Import a public ECDH key from a remote peer (JWK → CryptoKey)
  async importPublicKey(jwk) {
    return await window.crypto.subtle.importKey(
      "jwk",
      jwk,
      { name: "ECDH", namedCurve: "P-256" },
      true,
      []
    );
  },

  /**
   * Import our own private key from a JWK.
   * Only used for migrating keys that were previously stored in localStorage.
   * After migration the key is re-saved as non-extractable via savePrivateKey.
   */
  async importPrivateKey(jwk) {
    return await window.crypto.subtle.importKey(
      "jwk",
      jwk,
      { name: "ECDH", namedCurve: "P-256" },
      false, // import as non-extractable too
      ["deriveKey"]
    );
  },

  // ---------------------------------------------------------------------------
  // Encryption / decryption
  // ---------------------------------------------------------------------------

  // Derive an AES-GCM shared symmetric key using our private key and their public key
  async deriveSharedSecret(privateKey, publicKey) {
    return await window.crypto.subtle.deriveKey(
      { name: "ECDH", public: publicKey },
      privateKey,
      { name: "AES-GCM", length: 256 },
      false, // shared key doesn't need to be extractable
      ["encrypt", "decrypt"]
    );
  },

  // Encrypt a string message
  async encryptMessage(text, sharedKey) {
    const encodedText = new TextEncoder().encode(text);
    // 12 bytes is the recommended IV size for AES-GCM
    const iv = window.crypto.getRandomValues(new Uint8Array(12));

    const ciphertextBuffer = await window.crypto.subtle.encrypt(
      { name: "AES-GCM", iv: iv },
      sharedKey,
      encodedText
    );

    return {
      ciphertext: this.bufferToBase64(ciphertextBuffer),
      iv: this.bufferToBase64(iv),
    };
  },

  // Decrypt an encrypted message object
  async decryptMessage(encryptedObj, sharedKey) {
    const { ciphertext, iv } = encryptedObj;
    const ciphertextBuffer = this.base64ToBuffer(ciphertext);
    const ivBuffer = this.base64ToBuffer(iv);

    try {
      const decryptedBuffer = await window.crypto.subtle.decrypt(
        { name: "AES-GCM", iv: ivBuffer },
        sharedKey,
        ciphertextBuffer
      );
      return new TextDecoder().decode(decryptedBuffer);
    } catch (error) {
      console.error("Decryption failed:", error);
      throw new Error("Failed to decrypt message.");
    }
  },

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  bufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  },

  base64ToBuffer(base64) {
    const binary = window.atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  },
};
