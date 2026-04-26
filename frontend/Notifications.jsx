import { useEffect } from "react";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function useNotifications() {

  const fetchNotifications = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/notifications");
      const data = await res.json();

      if (data.success) {
        data.data.forEach((notif) => {
          toast.info(notif.message, {
            position: "top-right",
            autoClose: 4000,
          });
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