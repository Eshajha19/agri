import React, { useEffect, useMemo, useState } from "react";
import {
  FaShareAlt,
  FaWhatsapp,
  FaSms,
  FaCopy,
  FaGift,
  FaTrophy,
  FaUsers,
  FaMapMarkedAlt,
  FaUserPlus,
} from "react-icons/fa";
import { useSearchParams } from "react-router-dom";
import { useTheme } from "./ThemeContext";
import { auth, db, doc, getDoc, setDoc, updateDoc } from "./lib/firebase";
import { onAuthStateChanged } from "firebase/auth";
import "./ReferralHub.css";

export default function ReferralHub() {
  const { theme } = useTheme();
  const [searchParams] = useSearchParams();
  const [loading, setLoading] = useState(true);
  const [loadingRedeem, setLoadingRedeem] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [redeemCode, setRedeemCode] = useState(searchParams.get("ref") || "");
  const [data, setData] = useState(null);

  const progressPercent = useMemo(() => {
    const next = data?.milestones?.next;
    if (!next) {
      return 100;
    }
    const count = Number(data?.stats?.referralCount || 0);
    return Math.min(100, Math.round((count / next) * 100));
  }, [data]);

  const normalizeReferralCode = (code) => {
    if (!code) return "";
    return code.replace(/[^A-Z0-9]/g, "").toUpperCase();
  };

  const generateReferralCode = () => {
    const array = new Uint8Array(10);
    crypto.getRandomValues(array);
    const hash = Array.from(array, (b) => b.toString(16).padStart(2, "0")).join("");
    return `FS${hash.slice(0, 10).toUpperCase()}`;
  };

  const getReferralBadge = (count) => {
    if (count >= 10) return "Village Mentor";
    if (count >= 5) return "Community Champion";
    if (count >= 3) return "Seed Builder";
    if (count >= 1) return "First Harvester";
    return "Starter";
  };

  const fetchDashboard = async () => {
    setLoading(true);
    setError("");
    try {
      const currentUser = auth.currentUser;
      if (!currentUser) {
        setError("Please sign in to view your referral dashboard.");
        setLoading(false);
        return;
      }

      const uid = currentUser.uid;
      const userRef = doc(db, "users", uid);
      const userSnap = await getDoc(userRef);
      const userData = userSnap.exists() ? userSnap.data() : {};

      // Generate or retrieve referral code
      let referralCode = normalizeReferralCode(userData?.referralCode || "");
      if (!referralCode) {
        for (let attempt = 0; attempt < 5; attempt++) {
          referralCode = generateReferralCode();
          const codeRef = doc(db, "referral_codes", referralCode);
          const codeSnap = await getDoc(codeRef);
          if (!codeSnap.exists()) {
            await setDoc(codeRef, {
              uid,
              displayName: userData?.displayName || "Farmer",
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            });
            await updateDoc(userRef, {
              referralCode,
              referralCodeIssuedAt: new Date().toISOString(),
            });
            break;
          }
        }
      }

      // Compute stats
      const referralCount = Number(userData?.referralCount || 0);
      const referralPoints = Number(userData?.referralPoints || referralCount * 50);
      const referralBadge = userData?.referralBadge || getReferralBadge(referralCount);
      const community = userData?.villageName || userData?.village || userData?.address || "Unknown village";
      const unlockedPremium = referralCount >= 5;

      // Build referral link
      const baseUrl = window.location.origin;
      const referralLink = `${baseUrl}/login?ref=${referralCode}`;

      // Fetch referral history
      const historyRef = doc(db, "referrals_history", uid);
      const historySnap = await getDoc(historyRef);
      let history = [];
      if (historySnap.exists()) {
        const historyData = historySnap.data();
        history = historyData?.items || [];
      }

      // Compute milestones
      const milestones = [1, 3, 5, 10];
      const unlockedMilestones = milestones.filter((m) => referralCount >= m);
      const nextMilestone = milestones.find((m) => referralCount < m);

      // Fetch leaderboard
      const leaderboardRef = doc(db, "leaderboard", "referrals");
      const leaderboardSnap = await getDoc(leaderboardRef);
      let farmers = [];
      let villages = [];
      if (leaderboardSnap.exists()) {
        const lbData = leaderboardSnap.data();
        farmers = lbData?.farmers || [];
        villages = lbData?.villages || [];
      }

      setData({
        referralCode,
        referralLink,
        share: {
          whatsapp: `https://wa.me/?text=Join%20Fasal%20Saathi%20using%20my%20referral%20code%20${referralCode}%20-%20${referralLink}`,
          sms: `sms:?body=Join%20Fasal%20Saathi%20using%20my%20referral%20code%20${referralCode}%20-%20${referralLink}`,
        },
        stats: {
          referralCount,
          referralPoints,
          referralBadge,
          community,
          unlockedPremium,
        },
        milestones: {
          all: milestones,
          unlocked: unlockedMilestones,
          next: nextMilestone,
        },
        history,
        leaderboard: { farmers, villages },
      });
    } catch (err) {
      const msg = err?.message || "Unable to load referral dashboard. Please try again.";
      setError(msg);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        fetchDashboard();
      } else {
        setData(null);
        setLoading(false);
      }
    });
    return () => unsubscribe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const copyToClipboard = async (text) => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setSuccess("Copied to clipboard.");
      setTimeout(() => setSuccess(""), 2200);
    } catch {
      setError("Could not copy to clipboard.");
      setTimeout(() => setError(""), 2200);
    }
  };

  const shareViaWhatsApp = () => {
    const url = data?.share?.whatsapp || "";
    if (!url) return;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const shareViaSms = () => {
    const url = data?.share?.sms || "";
    if (!url) return;
    window.location.href = url;
  };

  const redeemReferral = async (event) => {
    event.preventDefault();
    if (!redeemCode.trim()) {
      setError("Enter a referral code first.");
      return;
    }

    setLoadingRedeem(true);
    setError("");
    setSuccess("");

    try {
      const currentUser = auth.currentUser;
      if (!currentUser) {
        setError("Please sign in to redeem a referral code.");
        setLoadingRedeem(false);
        return;
      }

      const inviteeUid = currentUser.uid;
      const normalizedCode = normalizeReferralCode(redeemCode.trim());

      // Look up the inviter
      const codeRef = doc(db, "referral_codes", normalizedCode);
      const codeSnap = await getDoc(codeRef);
      if (!codeSnap.exists()) {
        setError("Referral code not found.");
        setLoadingRedeem(false);
        return;
      }

      const inviterUid = codeSnap.data()?.uid;
      if (!inviterUid) {
        setError("Invalid referral code.");
        setLoadingRedeem(false);
        return;
      }
      if (inviterUid === inviteeUid) {
        setError("Self referral is not allowed.");
        setLoadingRedeem(false);
        return;
      }

      // Check if already redeemed
      const inviteeRef = doc(db, "users", inviteeUid);
      const inviteeSnap = await getDoc(inviteeRef);
      const inviteeData = inviteeSnap.exists() ? inviteeSnap.data() : {};
      if (inviteeData?.referredByUid || inviteeData?.referralRedeemedAt) {
        setError("Referral already redeemed for this account.");
        setLoadingRedeem(false);
        return;
      }

      const inviterRef = doc(db, "users", inviterUid);
      const inviterSnap = await getDoc(inviterRef);
      const inviterData = inviterSnap.exists() ? inviterSnap.data() : {};

      const createdAt = new Date().toISOString();
      const rewardPoints = 50;
      const newReferrerCount = Number(inviterData?.referralCount || 0) + 1;
      const newReferrerPoints = Number(inviterData?.referralPoints || 0) + rewardPoints;

      // Record the referral history for inviter
      const historyDocRef = doc(db, "referrals_history", inviterUid);
      const historyDocSnap = await getDoc(historyDocRef);
      let existingHistory = [];
      if (historyDocSnap.exists()) {
        existingHistory = historyDocSnap.data()?.items || [];
      }
      const newHistoryItem = {
        id: `${inviterUid}_${inviteeUid}`,
        inviteeUid,
        inviteeName: inviteeData?.displayName || "Farmer",
        inviteeLocation: inviteeData?.villageName || inviteeData?.village || "Unknown village",
        referralCode: normalizedCode,
        status: "redeemed",
        rewardPoints,
        createdAt,
      };
      await setDoc(historyDocRef, { items: [newHistoryItem, ...existingHistory] });

      // Update invitee
      await updateDoc(inviteeRef, {
        referredByUid: inviterUid,
        referredByCode: normalizedCode,
        referralRedeemedAt: createdAt,
      });

      // Update inviter
      await updateDoc(inviterRef, {
        referralCount: newReferrerCount,
        referralPoints: newReferrerPoints,
        referralBadge: getReferralBadge(newReferrerCount),
        premiumUnlocked: newReferrerCount >= 5,
        updatedAt: createdAt,
      });

      setSuccess("Referral redeemed successfully!");
      setRedeemCode("");
      await fetchDashboard();
    } catch (err) {
      const msg = err?.message || "Failed to redeem referral code.";
      setError(msg);
    } finally {
      setLoadingRedeem(false);
    }
  };

  return (
    <div className={`referral-page ${theme === "dark" ? "theme-dark" : ""}`}>
      <div className="referral-shell">
        <header className="referral-hero">
          <div className="hero-icon"><FaShareAlt /></div>
          <div>
            <h1>Farmer Referral and Village Sharing</h1>
            <p>Invite fellow farmers, grow your village network, and unlock rewards together.</p>
          </div>
        </header>

        {loading ? (
          <div className="referral-status">
            Loading your referral dashboard...
          </div>
        ) : error ? (
          <div className="referral-status error">
            <p>{error}</p>
            <button type="button" className="retry-btn" onClick={fetchDashboard}>
              Retry
            </button>
          </div>
        ) : !data ? (
          <div className="referral-status">
            No referral data available.
          </div>
        ) : (
          <>
            {(error || success) && (
              <div className={`referral-status ${error ? "error" : "success"}`}>
                {error || success}
              </div>
            )}

            <section className="referral-grid">
              <article className="panel code-panel">
                <h2><FaGift /> Your Referral Code</h2>
                <div className="code-row">
                  <span className="code-pill">{data?.referralCode || "-"}</span>
                  <button type="button" onClick={() => copyToClipboard(data?.referralCode)}>
                    <FaCopy /> Copy
                  </button>
                </div>
                <div className="link-row">
                  <input value={data?.referralLink || ""} readOnly aria-label="Referral link" />
                  <button type="button" onClick={() => copyToClipboard(data?.referralLink)}>
                    <FaCopy /> Copy Link
                  </button>
                </div>
                <div className="share-buttons">
                  <button type="button" className="wa" onClick={shareViaWhatsApp}>
                    <FaWhatsapp /> Share on WhatsApp
                  </button>
                  <button type="button" className="sms" onClick={shareViaSms}>
                    <FaSms /> Share via SMS
                  </button>
                </div>
              </article>

              <article className="panel stats-panel">
                <h2><FaTrophy /> Rewards</h2>
                <div className="stat-list">
                  <div>
                    <label>Referrals</label>
                    <strong>{data?.stats?.referralCount || 0}</strong>
                  </div>
                  <div>
                    <label>Points</label>
                    <strong>{data?.stats?.referralPoints || 0}</strong>
                  </div>
                  <div>
                    <label>Badge</label>
                    <strong>{data?.stats?.referralBadge || "Starter"}</strong>
                  </div>
                  <div>
                    <label>Village</label>
                    <strong>{data?.stats?.community || "Unknown village"}</strong>
                  </div>
                </div>
                <div className="progress-wrap">
                  <div className="progress-label">
                    Next milestone: {data?.milestones?.next ? `${data.milestones.next} referrals` : "All milestones unlocked"}
                  </div>
                  <div className="progress-track" aria-hidden="true">
                    <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
                  </div>
                </div>
                {data?.stats?.unlockedPremium && (
                  <p className="premium-chip">Premium referral rewards unlocked</p>
                )}
              </article>
            </section>

            <section className="referral-grid second">
              <article className="panel redeem-panel">
                <h2><FaUserPlus /> Have a Referral Code?</h2>
                <p>New farmers can redeem once to join a trusted network.</p>
                <form onSubmit={redeemReferral} className="redeem-form">
                  <input
                    value={redeemCode}
                    onChange={(e) => setRedeemCode(e.target.value.toUpperCase())}
                    placeholder="Enter referral code"
                    maxLength={32}
                    aria-label="Redeem referral code"
                  />
                  <button type="submit" disabled={loadingRedeem}>
                    {loadingRedeem ? "Redeeming..." : "Redeem Referral"}
                  </button>
                </form>
                <p className="nearby-cta">Invite nearby farmers and build village-level resilience.</p>
              </article>

              <article className="panel history-panel">
                <h2><FaUsers /> Referral History</h2>
                <div className="history-list">
                  {data?.history?.length ? data.history.map((item) => (
                    <div key={item.id} className="history-item">
                      <div>
                        <strong>{item.inviteeName || "Farmer"}</strong>
                        <span>{item.inviteeLocation || "Unknown village"}</span>
                      </div>
                      <div>
                        <strong>+{item.rewardPoints || 0} pts</strong>
                        <span>{item.status || "redeemed"}</span>
                      </div>
                    </div>
                  )) : <p className="empty">No successful referrals yet.</p>}
                </div>
              </article>
            </section>

            <section className="referral-grid third">
              <article className="panel leaderboard-panel">
                <h2><FaTrophy /> Top Farmers</h2>
                <div className="list-board">
                  {data?.leaderboard?.farmers?.length ? data.leaderboard.farmers.map((farmer, idx) => (
                    <div key={farmer.uid || idx} className="board-row">
                      <span>#{idx + 1} {farmer.displayName}</span>
                      <span>{farmer.referralCount} referrals</span>
                    </div>
                  )) : <p className="empty">Leaderboard will update as referrals grow.</p>}
                </div>
              </article>

              <article className="panel leaderboard-panel">
                <h2><FaMapMarkedAlt /> Village Rankings</h2>
                <div className="list-board">
                  {data?.leaderboard?.villages?.length ? data.leaderboard.villages.map((village, idx) => (
                    <div key={`${village.community}-${idx}`} className="board-row">
                      <span>#{idx + 1} {village.community}</span>
                      <span>{village.referrals} referrals</span>
                    </div>
                  )) : <p className="empty">No village rankings available yet.</p>}
                </div>
              </article>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
