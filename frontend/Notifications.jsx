import { useEffect, useRef } from "react";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import apiClient from "./lib/apiClient";
import { auth } from "./lib/firebase";

const MAX_TRACKED_NOTIFICATIONS = 1000;
const NOTIFICATION_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

export default function useNotifications() {
  const seenIdsRef = useRef(new Map());

  const eventStatsRef = useRef({
    processed: 0,
    duplicates: 0,
    snapshots: 0,
  });

  const lastSnapshotHashRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);

  const mountedRef = useRef(true);
  const requestIdRef = useRef(0);

  const cleanupSeenNotifications = () => {
    const now = Date.now();

    for (const [key, timestamp] of seenIdsRef.current.entries()) {
      if (now - timestamp > NOTIFICATION_TTL_MS) {
        seenIdsRef.current.delete(key);
      }
    }

    if (seenIdsRef.current.size > MAX_TRACKED_NOTIFICATIONS) {
      const oldestEntries = [...seenIdsRef.current.entries()]
        .sort((a, b) => a[1] - b[1]);

      const excess =
        seenIdsRef.current.size -
        MAX_TRACKED_NOTIFICATIONS;

      oldestEntries
        .slice(0, excess)
        .forEach(([key]) =>
          seenIdsRef.current.delete(key)
        );
    }
  };

  const generateSnapshotHash = (notifications) => {
    try {
      return JSON.stringify(
        notifications.map((item) => ({
          id: item.id,
          type: item.type,
          message: item.message,
        }))
      );
    } catch {
      return null;
    }
  };

  const markAndToast = (notif) => {
    if (!notif || !notif.message) return;

    cleanupSeenNotifications();

    const notificationKey =
      notif.id ??
      `${notif.type || "notification"}:${notif.time || ""}:${notif.message}`;

    if (seenIdsRef.current.has(notificationKey)) {
      eventStatsRef.current.duplicates += 1;
      return;
    }

    seenIdsRef.current.set(
      notificationKey,
      Date.now()
    );

    eventStatsRef.current.processed += 1;

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
    const apiBase =
      import.meta.env.VITE_API_BASE || window.location.origin;

    const url = new URL("/api/notifications/stream", apiBase);

    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";

    const token = await getIdToken();

    if (token) {
      url.searchParams.set("token", token);
    }

    return url.toString();
  };

  const fetchNotifications = async () => {
    const requestId = ++requestIdRef.current;

    try {
      const res = await apiClient.get("/api/notifications");
      const data = res?.data;

      if (
        mountedRef.current &&
        requestId === requestIdRef.current &&
        data?.success &&
        Array.isArray(data?.data)
      ) {
        const snapshotHash =
          generateSnapshotHash(data.data);

        if (
          snapshotHash &&
          snapshotHash === lastSnapshotHashRef.current
        ) {
          return;
        }

        lastSnapshotHashRef.current =
          snapshotHash;

        data.data.forEach(markAndToast);

        console.info(
          `[NOTIFICATIONS] processed=${eventStatsRef.current.processed} duplicates=${eventStatsRef.current.duplicates} tracked=${seenIdsRef.current.size}`
        );

      }
    } catch (err) {
      console.warn(
        "[Notifications] Failed to fetch notifications:",
        err?.message || err
      );
    }
  };

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      requestIdRef.current++;
      seenIdsRef.current.clear();

      lastSnapshotHashRef.current = null;

      eventStatsRef.current.processed = 0;
      eventStatsRef.current.duplicates = 0;
      eventStatsRef.current.snapshots = 0;
    };
  }, []);

  useEffect(() => {
    let websocket = null;
    let fallbackTimer = null;
    let cancelled = false;

    const startPollingFallback = () => {
      if (fallbackTimer || cancelled) return;

      fallbackTimer = setInterval(() => {
        fetchNotifications();
      }, 60000);
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
          const requestId = ++requestIdRef.current;

          try {
            const payload = JSON.parse(event.data);

            if (
              mountedRef.current &&
              requestId === requestIdRef.current
            ) {
              if (
                payload.type === "snapshot" &&
                Array.isArray(payload.data)
              ) {
                const snapshotHash =
                  generateSnapshotHash(payload.data);

                if (
                  snapshotHash !==
                  lastSnapshotHashRef.current
                ) {
                  lastSnapshotHashRef.current =
                    snapshotHash;

                  payload.data.forEach(markAndToast);

                  eventStatsRef.current.snapshots += 1;

                  console.info(
                    `[NOTIFICATIONS] snapshots=${eventStatsRef.current.snapshots} tracked=${seenIdsRef.current.size}`
                  );
                }
              }

              if (
                payload.type === "notification" &&
                payload.data
              ) {
                markAndToast(payload.data);
              }
            }
          } catch (parseError) {
            console.log(
              "Notification stream parse error:",
              parseError
            );
          }
        };

        websocket.onopen = () => {
          stopPollingFallback();
        };

        websocket.onerror = () => {
          reconnectAttemptsRef.current += 1;

          console.warn(
            `[NOTIFICATIONS] websocket error, reconnect=${reconnectAttemptsRef.current}`
          );

          startPollingFallback();
        };

        websocket.onclose = () => {
          if (!cancelled) {
            startPollingFallback();
          }
        };
      } catch (error) {
        console.log(
          "Notification websocket unavailable:",
          error
        );

        startPollingFallback();
      }
    };

    connectWebSocket();

    return () => {
      cancelled = true;

      stopPollingFallback();

      if (websocket) {
        websocket.onopen = null;
        websocket.onclose = null;
        websocket.onerror = null;
        websocket.onmessage = null;
        websocket.close();
      }
    };
  }, []);
}