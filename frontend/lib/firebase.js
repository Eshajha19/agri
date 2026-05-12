import { initializeApp, getApps } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore, doc, onSnapshot, getDoc, setDoc, updateDoc } from "firebase/firestore";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "",
};

const isConfigured = !!(
  firebaseConfig.apiKey &&
  firebaseConfig.apiKey.length > 10 &&
  firebaseConfig.projectId &&
  firebaseConfig.authDomain
);

let app  = null;
let auth = null;
let db   = null;

if (isConfigured) {
  try {
    app  = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
    auth = getAuth(app);
    db   = getFirestore(app);

    console.log("Firebase initialised", {
      projectId: firebaseConfig.projectId,
      authDomain: firebaseConfig.authDomain,
    });
  } catch (err) {
    console.error("Firebase initialisation failed:", err);
    // Leave auth/db as null so the rest of the app falls back gracefully
    app = auth = db = null;
  }
} else {
  console.warn("Firebase not configured — missing API key or projectId");
}

export { app, auth, db, doc, onSnapshot, getDoc, setDoc, updateDoc };
export const isFirebaseConfigured = () => isConfigured && auth !== null;
