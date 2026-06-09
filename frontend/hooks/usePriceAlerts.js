import { useEffect, useRef, useState, useCallback } from "react";
import { toast } from "react-toastify";
import { auth } from "../lib/firebase";

const WS_BASE_URL = import.meta.env.VITE_WS_BASE || "";
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000]; // exponential backoff caps at 8s
const HEARTBEAT_INTERVAL = 30000; // 30s
const MAX_MISSED_HEARTBEATS = 2;

/**
 * usePriceAlerts
 *
 * Manages a dedicated WebSocket connection for real-time price alerts.
 * Features:
 * - Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s max)
 * - Heartbeat/ping to detect stale connections on mobile networks
 * - delivery_ack sent for every received price_alert
 * - Connection status exposed for UI indicator
 * - Crop/region subscription management
 */
export default function usePriceAlerts() {
  const [status, setStatus] = useState("disconnected"); // "connected" | "connecting" | "disconnected" | "reconnecting"
  const [lastAlert, setLastAlert] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const heartbeatTimerRef = useRef(null);
  const missedHeartbeatsRef = useRef(0);
  const mountedRef = useRef(true);
  const subscribedCropsRef = useRef(new Set());
  const subscribedRegionsRef = useRef(new Set());

  const clearTimers = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
  }, []);

  const buildUrl = useCallback(async () => {
    const apiBase = WS_BASE_URL || window.location.origin;
    const url = new URL("/api/notifications/stream", apiBase);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";

    const token = await auth?.currentUser?.getIdToken();
    if (token) {
      url.searchParams.set("token", token);
    }

    // Add crop subscriptions
    const crops = Array.from(subscribedCropsRef.current);
    if (crops.length) {
      url.searchParams.set("crops", crops.join(","));
    }

    // Add region subscriptions
    const regions = Array.from(subscribedRegionsRef.current);
    if (regions.length) {
      url.searchParams.set("regions", regions.join(","));
    }

    return url.toString();
  }, []);

  const sendAck = useCallback((notificationId) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: "delivery_ack", notification_id: notificationId }));
    } catch (e) {
      console.warn("[PriceAlerts] Failed to send ack:", e);
    }
  }, []);

  const handleMessage = useCallback((event) => {
    try {
      const payload = JSON.parse(event.data);

      if (payload.type === "price_alert" && payload.data) {
        const alert = payload.data;
        setLastAlert(alert);

        // Send delivery acknowledgment
        if (payload.notification_id) {
          sendAck(payload.notification_id);
        }

        toast.info(alert.message, {
          position: "top-right",
          autoClose: 6000,
          icon: "📈",
        });
      }

      if (payload.type === "snapshot") {
        // Snapshot received — connection is healthy
        missedHeartbeatsRef.current = 0;
      }
    } catch (parseError) {
      console.warn("[PriceAlerts] Parse error:", parseError);
    }
  }, [sendAck]);

  const connect = useCallback(async () => {
    if (!mountedRef.current) return;
    if (!auth?.currentUser) {
      setStatus("disconnected");
      return;
    }

    clearTimers();
    setStatus("connecting");

    try {
      const url = await buildUrl();
      const ws = new WebSocket(url);

      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        reconnectAttemptRef.current = 0;
        missedHeartbeatsRef.current = 0;
        setStatus("connected");

        // Start heartbeat
        heartbeatTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            try {
              ws.send(JSON.stringify({ type: "ping" }));
              missedHeartbeatsRef.current += 1;
              if (missedHeartbeatsRef.current > MAX_MISSED_HEARTBEATS) {
                console.warn("[PriceAlerts] Missed heartbeats, forcing reconnect");
                ws.close(1000, "heartbeat_timeout");
              }
            } catch (e) {
              ws.close(1000, "heartbeat_error");
            }
          }
        }, HEARTBEAT_INTERVAL);
      };

      ws.onmessage = (event) => {
        // Reset missed heartbeats on any message
        missedHeartbeatsRef.current = 0;
        handleMessage(event);
      };

      ws.onerror = () => {
        // Let onclose handle reconnection logic
      };

      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        clearTimers();
        setStatus("reconnecting");

        const delay = RECONNECT_DELAYS[
          Math.min(reconnectAttemptRef.current, RECONNECT_DELAYS.length - 1)
        ];
        reconnectAttemptRef.current += 1;

        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, delay);
      };
    } catch (error) {
      console.error("[PriceAlerts] Connection error:", error);
      setStatus("disconnected");

      const delay = RECONNECT_DELAYS[
        Math.min(reconnectAttemptRef.current, RECONNECT_DELAYS.length - 1)
      ];
      reconnectAttemptRef.current += 1;

      reconnectTimerRef.current = setTimeout(() => {
        connect();
      }, delay);
    }
  }, [buildUrl, clearTimers, handleMessage]);

  const disconnect = useCallback(() => {
    mountedRef.current = false;
    clearTimers();
    const ws = wsRef.current;
    if (ws) {
      ws.onclose = null;
      ws.close();
      wsRef.current = null;
    }
  }, [clearTimers]);

  const subscribeCrops = useCallback((crops) => {
    subscribedCropsRef.current = new Set(crops);
    // Reconnect to apply new subscriptions
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "subscribe_crops", crops }));
      } catch (e) {
        console.warn("[PriceAlerts] Failed to update crop subscription:", e);
      }
    } else {
      reconnectAttemptRef.current = 0;
      connect();
    }
  }, [connect]);

  const subscribeRegions = useCallback((regions) => {
    subscribedRegionsRef.current = new Set(regions);
    reconnectAttemptRef.current = 0;
    connect();
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    status,
    lastAlert,
    subscribeCrops,
    subscribeRegions,
    reconnect: connect,
    disconnect,
  };
}