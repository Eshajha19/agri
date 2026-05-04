const { webcrypto } = require('crypto');

async function testECDH() {
  const keyPair = await webcrypto.subtle.generateKey(
    { name: "ECDH", namedCurve: "P-256" },
    true,
    ["deriveKey"]
  );

  try {
    const derivedKey = await webcrypto.subtle.deriveKey(
      { name: "ECDH", public: keyPair.publicKey },
      keyPair.privateKey,
      { name: "AES-GCM", length: 256 },
      false,
      ["encrypt", "decrypt"]
    );
    console.log("Success! Key derived.");
  } catch (err) {
    console.error("Failed to derive key against itself:", err);
  }
}

testECDH();
