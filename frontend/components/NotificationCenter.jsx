import React, { useState, useEffect } from "react";
import { db } from "../firebase"; // Firestore instance
import { collection, query, where, onSnapshot } from "firebase/firestore";

function NotificationCenter({ userId }) {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    const q = query(collection(db, "notifications"), where("userId", "==", userId));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      setNotifications(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() })));
    });
    return () => unsubscribe();
  }, [userId]);

  return (
    <div className="notification-center">
      <h2>Claim Notifications</h2>
      <ul>
        {notifications.map(n => (
          <li key={n.id} className={n.read ? "read" : "unread"}>
            <p><strong>{n.title}</strong></p>
            <p>{n.message}</p>
            <small>{new Date(n.createdAt.seconds * 1000).toLocaleString()}</small>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default NotificationCenter;
