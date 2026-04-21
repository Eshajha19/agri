import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAJP3Ko_khVAB-E0x5qAbqyE0ObpARebrA",
  authDomain: "fasal-saathi-fd996.firebaseapp.com",
  projectId: "fasal-saathi-fd996",
  storageBucket: "fasal-saathi-fd996.firebasestorage.app",
  messagingSenderId: "439991273459",
  appId: "1:439991273459:web:95bc25b6a92cdbe1c9b486",
  measurementId: "G-0FD1CCB5S9"
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);