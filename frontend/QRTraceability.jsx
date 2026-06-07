/**
 * QRTraceability.jsx
 *
 * Security fix: batch data is now stored and fetched from the backend
 * (POST /api/supply-chain/trace-batch to register,
 *  GET  /api/supply-chain/trace-batch/:id to fetch).
 *
 * Previously all batch data lived exclusively in localStorage, which meant:
 *  - Batches disappeared when the browser cache was cleared.
 *  - Any user could open DevTools and modify journey/status/certification
 *    data before sharing the QR code.
 *  - The "Verified Origin" badge shown to consumers was based on
 *    unverified client-side data.
 *  - Two hardcoded mock batches were shown as fallback, so consumers
 *    scanning old QR codes could see fabricated provenance data.
 *
 * Now:
 *  - Batch registration POSTs to the backend (requires auth).
 *  - The consumer-facing viewer GETs from the backend (public, read-only).
 *  - localStorage is used only to remember the list of batch IDs the
 *    farmer registered on this device (for the management list UI).
 *    The actual batch data always comes from the server.
 *  - No mock/fallback data is ever shown.
 */
import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { QRCodeSVG } from 'qrcode.react';
import {
  QrCode, Sprout, MapPin, Calendar, CheckCircle,
  ShieldCheck, ArrowRight, Share2, Download, MessageCircle, AlertCircle,
} from 'lucide-react';
import './QRTraceability.css';
import SoilChatbot from './SoilChatbot';
import apiClient from './lib/apiClient';

// Only the list of batch IDs is kept in localStorage — not the batch data.
const BATCH_IDS_KEY = 'qrFarmBatchIds_v2';
const MAX_BATCH_IDS = 100;

function loadBatchIds() {
  try {
    const raw = localStorage.getItem(BATCH_IDS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.slice(0, MAX_BATCH_IDS) : [];
  } catch {
    return [];
  }
}

function saveBatchIds(ids) {
  try {
    localStorage.setItem(BATCH_IDS_KEY, JSON.stringify(ids.slice(0, MAX_BATCH_IDS)));
  } catch {
    // localStorage quota exceeded — silently ignore; the IDs are a UI convenience only.
  }
}

export default function QRTraceability() {
  const { id: routeId } = useParams();

  // List of batch IDs registered on this device (UI convenience only).
  const [batchIds, setBatchIds] = useState(loadBatchIds);

  // Batches fetched from the server, keyed by ID.
  const [batchCache, setBatchCache] = useState({});

  const [selectedBatch, setSelectedBatch] = useState(null);
  const [viewMode, setViewMode] = useState('list'); // 'list' | 'viewer'
  const [showAdvisor, setShowAdvisor] = useState(false);

  // Per-batch loading/error state for the management list.
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState('');

  // Viewer-specific loading/error (used when navigating via QR URL).
  const [viewerLoading, setViewerLoading] = useState(false);
  const [viewerError, setViewerError] = useState('');

  // Form state.
  const [formError, setFormError] = useState('');
  const [formSubmitting, setFormSubmitting] = useState(false);

  // ── Fetch a single batch from the server ──────────────────────────────────
  const fetchBatch = useCallback(async (id) => {
    if (batchCache[id]) return batchCache[id];
    try {
      const res = await apiClient.get(`/api/supply-chain/trace-batch/${encodeURIComponent(id)}`);
      const batch = res.data?.batch;
      if (batch) {
        setBatchCache(prev => ({ ...prev, [id]: batch }));
      }
      return batch || null;
    } catch (err) {
      if (err?.response?.status === 404) return null;
      throw err;
    }
  }, [batchCache]);

  // ── Load all batches for the management list ──────────────────────────────
  useEffect(() => {
    if (batchIds.length === 0) return;
    let cancelled = false;
    setListLoading(true);
    setListError('');

    Promise.all(batchIds.map(id => fetchBatch(id).catch(() => null)))
      .then(() => { if (!cancelled) setListLoading(false); })
      .catch(() => { if (!cancelled) { setListLoading(false); setListError('Failed to load some batches.'); } });

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [batchIds]);

  // ── Handle direct QR URL navigation (/trace/:id) ─────────────────────────
  useEffect(() => {
    if (!routeId) return;
    let cancelled = false;
    setViewerLoading(true);
    setViewerError('');

    fetchBatch(routeId)
      .then(batch => {
        if (cancelled) return;
        if (batch) {
          setSelectedBatch(batch);
          setViewMode('viewer');
        } else {
          setViewerError('Batch not found. This QR code may be invalid or expired.');
        }
        setViewerLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          setViewerError('Failed to load batch data. Please try again.');
          setViewerLoading(false);
        }
      });

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeId]);

  // ── Register a new batch ──────────────────────────────────────────────────
  const generateNewBatch = async (e) => {
    e.preventDefault();
    setFormError('');
    setFormSubmitting(true);

    const formData = new FormData(e.target);
    const year = new Date().getFullYear();
    const ts = Date.now();
    const rand = Math.floor(Math.random() * 0xffff).toString(16).toUpperCase().padStart(4, '0');
    const newId = `BATCH-${year}-${ts}-${rand}`;

    const payload = {
      id: newId,
      crop: formData.get('crop'),
      variety: formData.get('variety'),
      harvestDate: formData.get('harvestDate'),
      farm: 'My Smart Farm',
      journey: [
        {
          date: new Date().toISOString().split('T')[0],
          event: 'Registration',
          location: 'My Smart Farm',
          details: 'Batch registered for traceability.',
        },
      ],
    };

    try {
      const res = await apiClient.post('/api/supply-chain/trace-batch', payload);
      const batch = res.data?.batch;
      if (batch) {
        setBatchCache(prev => ({ ...prev, [newId]: batch }));
        const updatedIds = [newId, ...batchIds];
        setBatchIds(updatedIds);
        saveBatchIds(updatedIds);
        e.target.reset();
      }
    } catch (err) {
      const status = err?.response?.status;
      if (status === 401) {
        setFormError('You must be logged in to register a batch.');
      } else if (status === 409) {
        setFormError('A batch with this ID already exists. Please try again.');
      } else {
        setFormError('Failed to register batch. Please try again.');
      }
    } finally {
      setFormSubmitting(false);
    }
  };

  const openBatch = async (id) => {
    try {
      const batch = await fetchBatch(id);
      if (batch) {
        setSelectedBatch(batch);
        setViewMode('viewer');
      }
    } catch {
      setListError('Failed to load batch details.');
    }
  };

  const shareQR = (batchId) => {
    const url = `${window.location.origin}/trace/${batchId}`;
    if (navigator.share) {
      navigator.share({ title: 'Trace My Produce', url });
    } else {
      navigator.clipboard?.writeText(url).catch(() => {});
      alert(`Share URL: ${url}`);
    }
  };

  // ── Consumer-facing viewer ────────────────────────────────────────────────
  if (viewerLoading) {
    return (
      <div className="trace-viewer">
        <div className="trace-header">
          <p>Loading batch data…</p>
        </div>
      </div>
    );
  }

  if (viewerError && routeId) {
    return (
      <div className="trace-viewer">
        <div className="trace-header">
          <AlertCircle size={20} style={{ color: '#c62828', marginRight: 8 }} />
          <p style={{ color: '#c62828' }}>{viewerError}</p>
        </div>
      </div>
    );
  }

  if (viewMode === 'viewer' && selectedBatch) {
    return (
      <div className="trace-viewer">
        <div className="trace-header">
          <button className="back-btn" onClick={() => setViewMode('list')}>← Back to Management</button>
          {selectedBatch.status === 'Verified' && (
            <div className="verified-badge"><ShieldCheck size={16} /> Verified Origin</div>
          )}
        </div>

        <div className="trace-content">
          <div className="produce-header">
            <h1>{selectedBatch.crop}</h1>
            <p className="variety-tag">{selectedBatch.variety}</p>
          </div>

          <div className="trace-card main-info">
            <div className="info-item">
              <MapPin size={18} />
              <div>
                <label>Origin Farm</label>
                <p>{selectedBatch.farm}</p>
              </div>
            </div>
            <div className="info-item">
              <Calendar size={18} />
              <div>
                <label>Harvest Date</label>
                <p>{selectedBatch.harvestDate}</p>
              </div>
            </div>
            <div className="info-item">
              <QrCode size={18} />
              <div>
                <label>Batch ID</label>
                <p>{selectedBatch.id}</p>
              </div>
            </div>
          </div>

          <div className="journey-section">
            <h3><Sprout size={20} /> Farm-to-Table Journey</h3>
            <div className="timeline">
              {(selectedBatch.journey || []).map((step, idx) => (
                <div key={idx} className="timeline-item">
                  <div className="timeline-marker"></div>
                  <div className="timeline-content">
                    <div className="timeline-header">
                      <span className="event">{step.event}</span>
                      <span className="date">{step.date}</span>
                    </div>
                    <p className="location">{step.location}</p>
                    <p className="details">{step.details}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <button className="advisor-fab" onClick={() => setShowAdvisor(true)} aria-label="Open AI Advisor">
          <MessageCircle size={24} />
        </button>
        {showAdvisor && (
          <div className="advisor-overlay" onClick={() => setShowAdvisor(false)}>
            <div className="advisor-modal" onClick={e => e.stopPropagation()}>
              <SoilChatbot onClose={() => setShowAdvisor(false)} />
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── Farmer management view ────────────────────────────────────────────────
  const managedBatches = batchIds
    .map(id => batchCache[id])
    .filter(Boolean);

  return (
    <div className="qr-farm-container">
      <div className="qr-farm-header">
        <h1><QrCode size={32} /> QR-Farm Traceability</h1>
        <p>Ensure transparency and build trust with your consumers.</p>
      </div>

      <div className="qr-farm-grid">
        <div className="registration-section">
          <h3>Register New Batch</h3>
          <form className="batch-form" onSubmit={generateNewBatch}>
            <div className="input-group">
              <label>Crop Name</label>
              <input name="crop" placeholder="e.g. Organic Tomatoes" required />
            </div>
            <div className="input-group">
              <label>Variety</label>
              <input name="variety" placeholder="e.g. Cherry Tomatoes" required />
            </div>
            <div className="input-group">
              <label>Harvest Date</label>
              <input name="harvestDate" type="date" required />
            </div>
            {formError && (
              <div className="error-msg" role="alert">
                <AlertCircle size={14} style={{ marginRight: 6 }} />{formError}
              </div>
            )}
            <button type="submit" className="generate-btn" disabled={formSubmitting}>
              {formSubmitting ? 'Registering…' : 'Generate Traceability ID'}
            </button>
          </form>
        </div>

        <div className="batches-section">
          <h3>Your Tracked Batches</h3>
          {listLoading && <p className="loading-msg">Loading batches…</p>}
          {listError && (
            <div className="error-msg" role="alert">
              <AlertCircle size={14} style={{ marginRight: 6 }} />{listError}
            </div>
          )}
          {!listLoading && managedBatches.length === 0 && (
            <p className="empty-msg">No batches registered yet. Use the form to create your first batch.</p>
          )}
          <div className="batch-list">
            {managedBatches.map(batch => (
              <div key={batch.id} className="batch-card" onClick={() => openBatch(batch.id)}>
                <div className="batch-qr">
                  <QRCodeSVG
                    value={batch.traceability?.verification_url_with_proof || `${window.location.origin}/trace/${batch.id}`}
                    size={80}
                    includeMargin={true}
                    level="H"
                  />
                </div>
                <div className="batch-info">
                  <h4>{batch.crop}</h4>
                  <p className="batch-id">{batch.id}</p>
                  <div className="batch-footer">
                    <span className="status-tag verified">{batch.status}</span>
                    <div className="batch-actions">
                      <button
                        className="test-link-btn"
                        onClick={(e) => { e.stopPropagation(); window.open(batch.traceability?.verification_url_with_proof || `/trace/${batch.id}`, '_blank'); }}
                      >
                        Test Link
                      </button>
                      <button className="view-link">
                        View Journey <ArrowRight size={14} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
