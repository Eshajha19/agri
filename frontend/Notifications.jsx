import { useEffect, useRef } from "react";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import apiClient from "./lib/apiClient";
import { auth } from "./lib/firebase";

export default function useNotifications() {
  const seenIdsRef = useRef(new Set());

  const markAndToast = (notif) => {
    if (!notif || !notif.message) return;

    const notificationKey =
      notif.id ?? `${notif.type || "notification"}:${notif.time || ""}:${notif.message}`;

    if (seenIdsRef.current.has(notificationKey)) return;

    seenIdsRef.current.add(notificationKey);
    toast.info(notif.message, {
      position: "top-right",
      autoClose: 4000,
    });
  };

const getIdToken = async () => {
    const user = auth?.currentUser;
    if (!user) return null;
    return user.getIdToken();
  };

  const buildStreamUrl = async () => {
    const apiBase = import.meta.env.VITE_API_BASE || window.location.origin;
    const url = new URL("/api/notifications/stream", apiBase);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const token = await getIdToken();
    if (token) {
      url.searchParams.set("token", token);
    }
    return url.toString();
  };

  const fetchNotifications = async () => {
    try {
      const res = await apiClient.get("/api/notifications");
      const data = res.data;

      if (data.success) {
        data.data.forEach(markAndToast);
      }
    } catch (err) {
      console.log("Notification fetch error:", err);
    }
  };

  useEffect(() => {
    let websocket = null;
    let fallbackTimer = null;
    let cancelled = false;

    const startPollingFallback = () => {
      if (fallbackTimer || cancelled) return;
      fallbackTimer = setInterval(fetchNotifications, 60000);
    };

    const stopPollingFallback = () => {
      if (fallbackTimer) {
        clearInterval(fallbackTimer);
        fallbackTimer = null;
      }
    };

    const connectWebSocket = async () => {
      if (!auth?.currentUser) {
        startPollingFallback();
        return;
      }


      fetchNotifications();

      if (typeof WebSocket === "undefined") {
        startPollingFallback();
        return;
      }

      try {
        const streamUrl = await buildStreamUrl();
        if (!streamUrl.includes("token=")) {
          startPollingFallback();
          return;
        }

        websocket = new WebSocket(streamUrl);

        websocket.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            if (payload.type === "snapshot" && Array.isArray(payload.data)) {
              payload.data.forEach(markAndToast);
            }
            if (payload.type === "notification" && payload.data) {
              markAndToast(payload.data);
            }
          } catch (parseError) {
            console.log("Notification stream parse error:", parseError);
          }
        };

        websocket.onopen = () => {
          stopPollingFallback();
        };

        websocket.onerror = () => {
          startPollingFallback();
        };

        websocket.onclose = () => {
          if (!cancelled) {
            startPollingFallback();
          }
        };
      } catch (error) {
        console.log("Notification websocket unavailable:", error);
        startPollingFallback();
      }
    };

    connectWebSocket();

    return () => {
      cancelled = true;
      stopPollingFallback();
      if (websocket) {
        websocket.close();
      }
    };
  }, []);
}
