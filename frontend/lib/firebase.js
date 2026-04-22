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

const isConfigured = !!firebaseConfig.apiKey && firebaseConfig.apiKey.length > 10;

let app = null;
let auth = null;
let db = null;

if (isConfigured) {
  if (getApps().length === 0) {
    app = initializeApp(firebaseConfig);
  } else {
    app = getApps()[0];
  }
  auth = getAuth(app);
  db = getFirestore(app);
  console.log("Firebase initialized with config:", { 
    projectId: firebaseConfig.projectId, 
    authDomain: firebaseConfig.authDomain 
  });
} else {
  console.warn("Firebase not configured - missing API key");
}

export { app, auth, db, doc, onSnapshot, getDoc, setDoc, updateDoc };
export const isFirebaseConfigured = () => isConfigured;
