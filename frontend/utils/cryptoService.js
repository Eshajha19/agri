/**
 * Secure E2EE Cryptography Service using Web Crypto API
 * Implements Elliptic Curve Diffie-Hellman (ECDH) and AES-GCM encryption
 */

export const cryptoService = {
  // Generate a new ECDH P-256 key pair
  async generateECDHKeyPair() {
    return await window.crypto.subtle.generateKey(
      { name: "ECDH", namedCurve: "P-256" },
      true, // extractable (so we can save to localStorage)
      ["deriveKey"]
    );
  },

  // Export a key to JWK format for storage or transmission
  async exportKey(key) {
    return await window.crypto.subtle.exportKey("jwk", key);
  },

  // Import a public ECDH key from a remote peer
  async importPublicKey(jwk) {
    return await window.crypto.subtle.importKey(
      "jwk",
      jwk,
      { name: "ECDH", namedCurve: "P-256" },
      true,
      []
    );
  },

  // Import our own private key from local storage
  async importPrivateKey(jwk) {
    return await window.crypto.subtle.importKey(
      "jwk",
      jwk,
      { name: "ECDH", namedCurve: "P-256" },
      true,
      ["deriveKey"]
    );
  },

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

    // Encode to base64 for easy transport/storage
    const ciphertextBase64 = this.bufferToBase64(ciphertextBuffer);
    const ivBase64 = this.bufferToBase64(iv);

    return {
      ciphertext: ciphertextBase64,
      iv: ivBase64
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

  // Helpers
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
  }
};
