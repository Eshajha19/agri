import React, { useState, useEffect } from "react";
import { formatDistanceToNow, format } from "date-fns";
import { FaClock } from "react-icons/fa";

export default function LastUpdated({ timestamp }) {
  const [timeAgo, setTimeAgo] = useState(() => {
    if (!timestamp) return "";
    try {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
    } catch {
      return "";
    }
  });

  useEffect(() => {
    if (!timestamp) return;

    const updateTime = () => {
      try {
        const date = new Date(timestamp);
        setTimeAgo(formatDistanceToNow(date, { addSuffix: true }));
      } catch {
        setTimeAgo("");
      }
    };

    updateTime();
    const interval = setInterval(updateTime, 60000);
    return () => clearInterval(interval);
  }, [timestamp]);

  if (!timestamp) return null;

  try {
    const date = new Date(timestamp);
    const formattedTime = format(date, "h:mm a");

    return (
      <div 
        className="last-updated" 
        style={{ 
          fontSize: '0.85rem', 
          color: 'var(--text-muted, #6b7280)', 
          display: 'flex', 
          alignItems: 'center', 
          gap: '6px', 
          marginTop: '8px',
          fontWeight: '500'
        }}
      >
        <FaClock />
        <span title={`Fetched at: ${formattedTime}`}>
          Last Updated: {timeAgo}
        </span>
      </div>
    );
  } catch {
    return null;
  }
}
