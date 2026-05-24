import React, { useEffect, useMemo, useState } from "react";
import { JitsiMeeting } from "@jitsi/react-sdk";
import "./ExpertDirectory.css";
import { db, auth } from "../lib/firebase";
import { collection, getDocs, updateDoc, doc, query, where } from "firebase/firestore";
import { toast } from "react-toastify";
import {
  Video,
  PhoneOff,
  MessageSquare,
  Clock,
  ShieldCheck,
  Users,
  LoaderCircle,
} from "lucide-react";

const JITSI_DOMAIN = "meet.jit.si";

const slugifyRoomName = (value) =>
  String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);

const buildRoomName = (consultation) => {
  const baseSeed = consultation?.roomName || consultation?.meetingRoom || consultation?.id || consultation?.expertName || "live-expert-consultation";
  const randomSuffix = typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID().slice(0, 8)
    : Math.random().toString(36).slice(2, 10);

  return slugifyRoomName(`fasal-saathi-${baseSeed}-${randomSuffix}`) || `fasal-saathi-live-${Date.now()}`;
};

function TeleConsultation({ userData, consultation, onEnd }) {
  const [callDuration, setCallDuration] = useState(0);
  const [isMeetingReady, setIsMeetingReady] = useState(false);
  const [meetingError, setMeetingError] = useState("");
  const [notes, setNotes] = useState("");
  const [showNotes, setShowNotes] = useState(false);
  const [isSavingEndState, setIsSavingEndState] = useState(false);

  const roomName = useMemo(() => buildRoomName(consultation), [consultation?.roomName, consultation?.meetingRoom, consultation?.id, consultation?.expertName]);
  const expertName = consultation?.expertName || "Live Expert Consultation";
  const sessionLabel = consultation?.expertSpecialization || "Crop guidance, soil analysis, fertilizer recommendations, and disease diagnosis";
  const farmerName = userData?.displayName || userData?.name || userData?.fullName || "Farmer";

  useEffect(() => {
    const timer = setInterval(() => {
      setCallDuration((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    setCallDuration(0);
    setIsMeetingReady(false);
    setMeetingError("");
    setShowNotes(false);
    setNotes("");
  }, [roomName]);

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const handleMeetingReady = () => {
    setIsMeetingReady(true);
    toast.success("Connected to expert");
  };

  const handleEndCall = async () => {
    setIsSavingEndState(true);
    try {
      const consultationsRef = collection(db, "consultations");

      // If the consultation object already carries its Firestore document ID,
      // do a direct point-lookup — no query needed, no risk of touching another
      // user's record.
      if (consultation.id) {
        const docRef = doc(db, "consultations", consultation.id);
        await updateDoc(docRef, {
          status: "completed",
          duration: callDuration,
          endTime: new Date().toISOString(),
          notes: notes,
        });
      } else {
        // Fallback: scope the query to BOTH the expert AND the current user so
        // we never accidentally update another farmer's consultation record.
        // Previously the query only filtered on expertId, meaning snapshot.docs[0]
        // could be any farmer's record for that expert.
        const currentUid = auth?.currentUser?.uid || userData?.uid;
        const constraints = [
          where("expertId", "==", consultation.expertId),
          where("status", "==", "scheduled"),
        ];
        if (currentUid) {
          constraints.push(where("userId", "==", currentUid));
        }
        const q = query(consultationsRef, ...constraints);
        const snapshot = await getDocs(q);

        if (!snapshot.empty) {
          // Update only the first matching record that belongs to this user
          const docRef = snapshot.docs[0].ref;
          await updateDoc(docRef, {
            status: "completed",
            duration: callDuration,
            endTime: new Date().toISOString(),
            notes: notes,
          });
        }
      }

      toast.success("Consultation completed!");
      onEnd?.();
    } catch (error) {
      console.error("Error ending call:", error);
      toast.error("Failed to end consultation");
    } finally {
      setIsSavingEndState(false);
    }
  };

  return (
    <div className="tele-consultation-container tele-consultation-shell">
        <div className="call-header">
          <div className="expert-info">
            <img src={consultation.avatar || "https://randomuser.me/api/portraits/men/32.jpg"} alt="Expert" className="expert-video-avatar" />
            <div>
              <h3>{expertName}</h3>
              <p>{isMeetingReady ? "Connected" : "Connecting..."}</p>
            </div>
          </div>
          <div className="call-timer">
            <Clock size={16} />
            <span>{formatDuration(callDuration)}</span>
          </div>
          <button className="close-call-btn" onClick={handleEndCall} disabled={isSavingEndState}>
            <PhoneOff size={18} />
            <span>Leave</span>
          </button>
        </div>

        <div className="tele-consultation-hero">
          <div className="tele-consultation-copy">
            <span className="tele-consultation-badge">Live Expert Consultation</span>
            <h3>{expertName}</h3>
            <p>{sessionLabel}</p>
            <div className="tele-consultation-meta">
              <span><ShieldCheck size={14} /> Secure Jitsi room</span>
              <span><MessageSquare size={14} /> Chat enabled</span>
              <span><Users size={14} /> {farmerName}</span>
            </div>
          </div>
          <div className="tele-consultation-illustration" aria-hidden="true">
            <Video size={48} />
          </div>
        </div>

        <div className="tele-consultation-stage">
          {!isMeetingReady && !meetingError && (
            <div className="tele-consultation-loading">
              <div className="spinner"></div>
              <h4>Starting secure video room...</h4>
              <p>Connecting you to an agriculture expert.</p>
            </div>
          )}

          {meetingError ? (
            <div className="tele-consultation-error">
              <LoaderCircle size={40} />
              <h4>Unable to load the meeting room.</h4>
              <p>{meetingError}</p>
              <button className="save-notes-btn" onClick={handleEndCall} disabled={isSavingEndState}>
                Leave Consultation
              </button>
            </div>
          ) : (
            <JitsiMeeting
              domain={JITSI_DOMAIN}
              roomName={roomName}
              userInfo={{ displayName: farmerName }}
              configOverwrite={{
                prejoinPageEnabled: false,
                startWithAudioMuted: false,
                startWithVideoMuted: false,
                disableDeepLinking: true,
                toolbarButtons: [
                  "microphone",
                  "camera",
                  "chat",
                  "tileview",
                  "raisehand",
                  "desktop",
                  "fullscreen",
                  "hangup",
                ],
              }}
              interfaceConfigOverwrite={{
                DEFAULT_BACKGROUND: "#0f172a",
                MOBILE_APP_PROMO: false,
                SHOW_BRAND_WATERMARK: false,
                SHOW_JITSI_WATERMARK: false,
                SHOW_WATERMARK_FOR_GUESTS: false,
                TOOLBAR_ALWAYS_VISIBLE: true,
              }}
              onApiReady={() => handleMeetingReady()}
              onReadyToClose={handleEndCall}
              getIFrameRef={(iframeRef) => {
                if (iframeRef) {
                  iframeRef.style.width = "100%";
                  iframeRef.style.height = "100%";
                  iframeRef.style.border = "0";
                  iframeRef.title = `${expertName} consultation room`;
                }
              }}
            />
          )}
        </div>

        <div className="call-status-bar consultation-footer">
          <div className="status-item">
            <ShieldCheck size={14} />
            <span>Secure room</span>
          </div>
          <div className="status-item">
            <MessageSquare size={14} />
            <span>Chat available</span>
          </div>
          <div className="status-item">
            <Clock size={14} />
            <span>Room: {roomName}</span>
          </div>
        </div>
    </div>
  );
}

export default TeleConsultation;