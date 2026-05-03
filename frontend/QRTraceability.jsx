
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { QRCodeSVG } from 'qrcode.react';
import { QrCode, Sprout, MapPin, Calendar, CheckCircle, ShieldCheck, ArrowRight, Share2, Download, MessageCircle } from 'lucide-react';
import './QRTraceability.css';
import SoilChatbot from './SoilChatbot';

const MOCK_BATCHES = [
  {
    id: 'BATCH-2026-001',
    crop: 'Organic Basmati Rice',
    variety: 'Pusa 1121',
    harvestDate: '2026-04-15',
    farm: 'Green Valley Farms, Punjab',
    status: 'Verified',
    journey: [
      { date: '2026-01-10', event: 'Sowing', location: 'Green Valley Farms', details: 'Certified organic seeds used.' },
      { date: '2026-03-05', event: 'Quality Inspection', location: 'Agri-Gov Lab', details: 'Zero pesticide residue found.' },
      { date: '2026-04-15', event: 'Harvesting', location: 'Green Valley Farms', details: 'Hand-harvested at peak maturity.' },
      { date: '2026-04-20', event: 'Packaging', location: 'Fasal Saathi Hub', details: 'Eco-friendly vacuum packaging.' }
    ]
  },
  {
    id: 'BATCH-2026-002',
    crop: 'Alphonso Mangoes',
    variety: 'Ratnagiri',
    harvestDate: '2026-05-01',
    farm: 'Ratna Orchards, Maharashtra',
    status: 'Verified',
    journey: [
      { date: '2025-12-01', event: 'Flowering Stage', location: 'Ratna Orchards', details: 'Natural pollination by honeybees.' },
      { date: '2026-04-20', event: 'Ripening Control', location: 'Ratna Orchards', details: 'Ethylene-free natural ripening.' },
      { date: '2026-05-01', event: 'Harvesting', location: 'Ratna Orchards', details: 'Selected for export grade.' }
    ]
  }
];

export default function QRTraceability() {
  const { id: routeId } = useParams();
  const [batches, setBatches] = useState(() => {
    const saved = localStorage.getItem('qrFarmBatches');
    return saved ? JSON.parse(saved) : MOCK_BATCHES;
  });
  const [selectedBatch, setSelectedBatch] = useState(null);
  const [viewMode, setViewMode] = useState('list'); // 'list', 'details', 'viewer'
  const [showAdvisor, setShowAdvisor] = useState(false);

  useEffect(() => {
    if (routeId) {
      const found = batches.find(b => b.id === routeId);
      if (found) {
        setSelectedBatch(found);
        setViewMode('viewer');
      }
    }
  }, [routeId, batches]);

  useEffect(() => {
    localStorage.setItem('qrFarmBatches', JSON.stringify(batches));
  }, [batches]);

  const generateNewBatch = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const newBatch = {
      id: `BATCH-2026-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
      crop: formData.get('crop'),
      variety: formData.get('variety'),
      harvestDate: formData.get('harvestDate'),
      farm: 'My Smart Farm',
      status: 'Pending Verification',
      journey: [
        { date: new Date().toISOString().split('T')[0], event: 'Registration', location: 'My Smart Farm', details: 'Batch registered for traceability.' }
      ]
    };
    setBatches([newBatch, ...batches]);
    e.target.reset();
  };

  const shareQR = (batchId) => {
    const url = `${window.location.origin}/trace/${batchId}`;
    if (navigator.share) {
      navigator.share({
        title: 'Trace My Produce',
        text: `Trace the journey of our ${selectedBatch.crop} from farm to table.`,
        url: url
      });
    } else {
      alert(`URL copied to clipboard: ${url}`);
    }
  };

  if (viewMode === 'viewer' && selectedBatch) {
    return (
      <div className="trace-viewer">
        <div className="trace-header">
          <button className="back-btn" onClick={() => setViewMode('list')}>← Back to Management</button>
          <div className="verified-badge"><ShieldCheck size={16} /> Verified Origin</div>
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
              {selectedBatch.journey.map((step, idx) => (
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

        {/* Advisor Button and Modal */}
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
            <button type="submit" className="generate-btn">Generate Traceability ID</button>
          </form>
        </div>

        <div className="batches-section">
          <h3>Your Tracked Batches</h3>
          <div className="batch-list">
            {batches.map(batch => (
              <div key={batch.id} className="batch-card" onClick={() => { setSelectedBatch(batch); setViewMode('viewer'); }}>
                <div className="batch-qr">
                  <QRCodeSVG 
                    value={`${window.location.origin}/trace/${batch.id}`} 
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
                      <button className="test-link-btn" onClick={(e) => { e.stopPropagation(); window.open(`/trace/${batch.id}`, '_blank'); }}>
                        Test Link
                      </button>
                      <button className="view-link">View Journey <ArrowRight size={14} /></button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {selectedBatch && viewMode === 'details' && (
        <div className="qr-modal-overlay" onClick={() => setViewMode('list')}>
          <div className="qr-print-card" onClick={e => e.stopPropagation()}>
            <h2>QR Traceability Label</h2>
            <div className="print-qr-container">
              <QRCodeSVG value={`${window.location.origin}/trace/${selectedBatch.id}`} size={200} />
            </div>
            <div className="print-details">
              <h3>{selectedBatch.crop}</h3>
              <p>Scan to verify farm origin and journey.</p>
              <div className="print-meta">
                <span><strong>Farm:</strong> {selectedBatch.farm}</span>
                <span><strong>ID:</strong> {selectedBatch.id}</span>
              </div>
            </div>
            <div className="print-actions">
              <button onClick={() => window.print()} className="print-btn"><Download size={18} /> Save for Printing</button>
              <button onClick={() => shareQR(selectedBatch.id)} className="share-btn"><Share2 size={18} /> Share Link</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}