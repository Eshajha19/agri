import { useEffect, useRef } from "react";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function useNotifications() {
  const shownIds = useRef(new Set());

  const fetchNotifications = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/notifications");
      const data = await res.json();

      if (data.success) {
        data.data.forEach((notif) => {
          // Use a combination of ID and type to create a unique key since IDs might overlap between static and dynamic
          const uniqueKey = `${notif.type}-${notif.id}-${notif.message}`;
          
          if (!shownIds.current.has(uniqueKey)) {
            toast.info(notif.message, {
              position: "top-right",
              autoClose: 4000,
            });
            shownIds.current.add(uniqueKey);
          }
        });
      }
    } catch (err) {
      console.log("Notification fetch error:", err);
    }
  };

  useEffect(() => {
    fetchNotifications();

    // polling every 60 seconds
    const interval = setInterval(fetchNotifications, 60000);

    return () => clearInterval(interval);
  }, []);
}