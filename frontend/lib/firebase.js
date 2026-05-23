import { initializeApp, getApps } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore, doc, onSnapshot, getDoc, setDoc, updateDoc } from "firebase/firestore";

let app = null;
let auth = null;
let db = null;
let runtimeConfig = null;
let initPromise = null;

function getBuildTimeConfig() {
  return {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "",
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "",
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "",
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "",
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "",
    appId: import.meta.env.VITE_FIREBASE_APP_ID || "",
  };
}

function isConfigValid(config) {
  return !!(
    config &&
    config.apiKey &&
    config.apiKey.length > 10 &&
    config.projectId &&
    config.authDomain
  );
}

export async function fetchRuntimeConfig() {
  if (runtimeConfig !== null) return runtimeConfig;

  try {
    const response = await fetch("/api/firebase-config");
    if (response.ok) {
      runtimeConfig = await response.json();
    } else {
      runtimeConfig = null;
    }
  } catch (err) {
    console.warn("Could not fetch Firebase config from backend:", err);
    runtimeConfig = null;
  }
  return runtimeConfig;
}

export async function initializeFirebase() {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    // Try build-time config first (local dev)
    const buildConfig = getBuildTimeConfig();
    let config = isConfigValid(buildConfig) ? buildConfig : null;

    // Fall back to runtime config (production Docker)
    if (!config) {
      const runtime = await fetchRuntimeConfig();
      if (runtime && isConfigValid(runtime)) {
        config = runtime;
      }
    }

    if (!config) {
      console.warn("Firebase not configured — missing API key or projectId");
      return;
    }

    try {
      app = getApps().length === 0 ? initializeApp(config) : getApps()[0];
      auth = getAuth(app);
      db = getFirestore(app);

      console.log("Firebase initialised", {
        projectId: config.projectId,
        authDomain: config.authDomain,
      });
    } catch (err) {
      console.error("Firebase initialisation failed:", err);
      app = auth = db = null;
    }
  })();

  return initPromise;
}

export { app, auth, db, doc, onSnapshot, getDoc, setDoc, updateDoc };
export const isFirebaseConfigured = () => auth !== null;