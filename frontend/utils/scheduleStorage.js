// frontend/src/utils/scheduleStorage.js
import { db } from "../firebase";
import { collection, addDoc, getDocs, deleteDoc, doc } from "firebase/firestore";

const schedulesRef = (userId) => collection(db, "users", userId, "schedules");

export async function saveSchedule(userId, schedule) {
  await addDoc(schedulesRef(userId), schedule);
}

export async function getSchedules(userId) {
  const snapshot = await getDocs(schedulesRef(userId));
  return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
}

export async function deleteSchedule(userId, scheduleId) {
  await deleteDoc(doc(db, "users", userId, "schedules", scheduleId));
}
