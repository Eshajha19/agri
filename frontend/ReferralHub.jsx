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
import apiClient from "./lib/apiClient";
import { useTheme } from "./ThemeContext";
import "./ReferralHub.css";

const EMPTY_DATA = {
  referralCode: "",
  referralLink: "",
  share: { whatsapp: "", sms: "" },
  stats: {
    referralCount: 0,
    referralPoints: 0,
    referralBadge: "Starter",
    community: "Unknown village",
    unlockedPremium: false,
  },
  milestones: { all: [1, 3, 5, 10], unlocked: [], next: 1 },
  history: [],
  leaderboard: { farmers: [], villages: [] },
};

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

  const fetchDashboard = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.get("/api/referrals/dashboard");
      setData(res?.data?.data || null);
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        "Unable to load referral dashboard. Please sign in and try again.";

      setError(msg);

      setData(null);

    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
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
      const res = await apiClient.post("/api/referrals/redeem", {
        referral_code: redeemCode.trim(),
      });
      setSuccess(res?.data?.message || "Referral redeemed successfully.");
      setRedeemCode("");
      await fetchDashboard();
    } catch (err) {
      const msg = err?.response?.data?.detail || "Failed to redeem referral code.";
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

            <button
              type="button"
              className="retry-btn"
              onClick={fetchDashboard}
            >
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
