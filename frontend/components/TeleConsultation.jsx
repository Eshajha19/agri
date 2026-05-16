import React, { useState, useEffect } from "react";
import "./ExpertDirectory.css";
import { db } from "../lib/firebase";
import { collection, getDocs, updateDoc, doc, query, where } from "firebase/firestore";
import { toast } from "react-toastify";
import {
  X,
  Video,
  Phone,
  Mic,
  MicOff,
  Camera,
  CameraOff,
  PhoneOff,
  MessageSquare,
  Clock,
  User,
  CheckCircle,
} from "lucide-react";

function TeleConsultation({ consultation, onEnd }) {
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isMicEnabled, setIsMicEnabled] = useState(true);
  const [callDuration, setCallDuration] = useState(0);
  const [callStatus, setCallStatus] = useState("connecting");
  const [showNotes, setShowNotes] = useState(false);
  const [notes, setNotes] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setCallDuration((prev) => prev + 1);
    }, 1000);

    setTimeout(() => {
      setCallStatus("connected");
      toast.success("Connected to expert");
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const handleEndCall = async () => {
    setIsLoading(true);
    try {
      const consultationsRef = collection(db, "consultations");
      const q = query(consultationsRef, where("expertId", "==", consultation.expertId));
      const snapshot = await getDocs(q);

      if (!snapshot.empty) {
        const docRef = snapshot.docs[0].ref;
        await updateDoc(docRef, {
          status: "completed",
          duration: callDuration,
          endTime: new Date().toISOString(),
          notes: notes,
        });
      }

      toast.success("Consultation completed!");
      onEnd?.();
    } catch (error) {
      console.error("Error ending call:", error);
      toast.error("Failed to end consultation");
    } finally {
      setIsLoading(false);
    }
  };

  const toggleVideo = () => setIsVideoEnabled(!isVideoEnabled);
  const toggleMic = () => setIsMicEnabled(!isMicEnabled);

  return (
    <div className="tele-consultation-container">
        <div className="call-header">
          <div className="expert-info">
            <img src={consultation.avatar || "https://randomuser.me/api/portraits/men/32.jpg"} alt="Expert" className="expert-video-avatar" />
            <div>
              <h3>{consultation.expertName}</h3>
              <p>{callStatus === "connecting" ? "Connecting..." : callStatus === "connected" ? "Connected" : "Call Ended"}</p>
            </div>
          </div>
          <div className="call-timer">
            <Clock size={16} />
            <span>{formatDuration(callDuration)}</span>
          </div>
          <button className="close-call-btn" onClick={handleEndCall} disabled={isLoading}>
            <X size={20} />
          </button>
        </div>

        <div className="video-area">
          <div className="video-placeholder">
            {isVideoEnabled ? (
              <div className="video-enabled">
                <User size={64} />
                <p>Camera would show here</p>
                <p className="demo-note">(Demo Mode - Video integration available)</p>
              </div>
            ) : (
              <div className="video-disabled">
                <CameraOff size={64} />
                <p>Camera is off</p>
              </div>
            )}
          </div>

          <div className="self-view">
            {isVideoEnabled ? (
              <div className="self-video">
                <User size={32} />
              </div>
            ) : (
              <div className="self-video-off">
                <CameraOff size={24} />
              </div>
            )}
          </div>

          {callStatus === "connecting" && (
            <div className="connecting-overlay">
              <div className="spinner"></div>
              <p>Connecting to {consultation.expertName}...</p>
            </div>
          )}
        </div>

        <div className="call-controls">
          <button
            className={`control-btn ${!isMicEnabled ? "disabled" : ""}`}
            onClick={toggleMic}
            title={isMicEnabled ? "Mute" : "Unmute"}
          >
            {isMicEnabled ? <Mic size={24} /> : <MicOff size={24} />}
          </button>

          <button
            className={`control-btn ${!isVideoEnabled ? "disabled" : ""}`}
            onClick={toggleVideo}
            title={isVideoEnabled ? "Turn off camera" : "Turn on camera"}
          >
            {isVideoEnabled ? <Camera size={24} /> : <CameraOff size={24} />}
          </button>

          <button
            className={`control-btn ${showNotes ? "active" : ""}`}
            onClick={() => setShowNotes(!showNotes)}
            title="Notes"
          >
            <MessageSquare size={24} />
          </button>

          <button
            className="control-btn end-call"
            onClick={handleEndCall}
            disabled={isLoading}
            title="End call"
          >
            <PhoneOff size={24} />
          </button>
        </div>

        {showNotes && (
          <div className="notes-panel">
            <h4>Consultation Notes</h4>
            <textarea
              placeholder="Add notes about this consultation..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={4}
            />
            <button onClick={() => setShowNotes(false)} className="save-notes-btn">
              Save Notes
            </button>
          </div>
        )}

        <div className="call-status-bar">
          <div className="status-item">
            <CheckCircle size={14} />
            <span>Secure Connection</span>
          </div>
          <div className="status-item">
            <Video size={14} />
            <span>{isVideoEnabled ? "Video On" : "Video Off"}</span>
          </div>
          <div className="status-item">
            <Phone size={14} />
            <span>{isMicEnabled ? "Mic On" : "Mic Off"}</span>
          </div>
        </div>
    </div>
  );
}

export default TeleConsultation;