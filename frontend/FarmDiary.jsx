import React, { useState, useEffect } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { toast } from 'react-toastify';
import {
  Book, Plus, Download, Trash2, Edit2, Calendar, Clock,
  Check, X, Droplets, Sprout, Tractor, Activity, MessageCircle
} from 'lucide-react';
import './FarmDiary.css';
import SoilChatbot from './SoilChatbot';

const ACTIVITY_TYPES = ['Sowing', 'Irrigation', 'Fertilizer', 'Harvest', 'Pesticide', 'Other'];

export default function FarmDiary({ onClose }) {
  const [entries, setEntries] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({
    date: new Date().toISOString().split('T')[0],
    activityType: 'Sowing',
    notes: '',
    cost: '',
    reminderDate: '',
    isCompleted: true
  });
  const [showAdvisor, setShowAdvisor] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('fasalSaathiDiary');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        parsed.sort((a, b) => new Date(b.date) - new Date(a.date));
        setEntries(parsed);
      } catch (e) {
        console.error('Error loading diary entries', e);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('fasalSaathiDiary', JSON.stringify(entries));
  }, [entries]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({ ...prev, [name]: type === 'checkbox' ? checked : value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.date || !formData.notes || !formData.activityType) {
      toast.error('Please fill in required fields (Date, Type, Notes)');
      return;
    }
    if (editingId) {
      setEntries(entries.map(entry => entry.id === editingId ? { ...formData, id: editingId } : entry));
      toast.success('Entry updated successfully!');
    } else {
      const newEntry = { ...formData, id: Date.now().toString() };
      setEntries([newEntry, ...entries].sort((a, b) => new Date(b.date) - new Date(a.date)));
      toast.success('New activity logged!');
    }
    resetForm();
  };

  const handleEdit = (entry) => {
    setFormData(entry);
    setEditingId(entry.id);
    setShowForm(true);
  };

  const handleDelete = (id) => {
    if (window.confirm('Are you sure you want to delete this log?')) {
      setEntries(entries.filter(e => e.id !== id));
      toast.success('Entry deleted');
    }
  };

  const toggleStatus = (id) => {
    setEntries(entries.map(e => {
      if (e.id === id) {
        const isNowCompleted = !e.isCompleted;
        toast.info(isNowCompleted ? 'Task marked as completed' : 'Task marked as pending');
        return { ...e, isCompleted: isNowCompleted };
      }
      return e;
    }));
  };

  const resetForm = () => {
    setFormData({
      date: new Date().toISOString().split('T')[0],
      activityType: 'Sowing',
      notes: '',
      cost: '',
      reminderDate: '',
      isCompleted: true
    });
    setEditingId(null);
    setShowForm(false);
  };

  const generatePDF = () => {
    if (entries.length === 0) { toast.warning('No entries to export'); return; }
    try {
      const doc = new jsPDF();
      doc.setFontSize(20);
      doc.setTextColor(46, 204, 113);
      doc.text('Fasal Saathi - Farm Diary Report', 14, 22);
      doc.setFontSize(11);
      doc.setTextColor(100);
      doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 14, 30);
      doc.text(`Total Entries: ${entries.length}`, 14, 36);
      const totalCost = entries.reduce((sum, entry) => sum + (parseFloat(entry.cost) || 0), 0);
      doc.text(`Total Expense: ₹${totalCost.toFixed(2)}`, 14, 42);
      const tableRows = entries.map(entry => [
        entry.date, entry.activityType,
        entry.isCompleted ? 'Completed' : 'Pending',
        entry.notes, entry.cost || '-', entry.reminderDate || '-'
      ]);
      autoTable(doc, {
        head: [["Date", "Activity", "Status", "Notes", "Cost (₹)", "Reminder"]],
        body: tableRows, startY: 50, theme: 'grid',
        styles: { fontSize: 9, cellPadding: 3 },
        headStyles: { fillColor: [46, 204, 113], textColor: 255 },
        alternateRowStyles: { fillColor: [245, 255, 245] }
      });
      doc.save(`Farm_Diary_Report_${new Date().toISOString().split('T')[0]}.pdf`);
      toast.success('PDF Report downloaded successfully!');
    } catch (error) {
      console.error("PDF Export Error: ", error);
      toast.error('Failed to generate PDF. Please try again.');
    }
  };

  const today = new Date().toISOString().split('T')[0];
  const getActivityIcon = (type) => {
    switch (type) {
      case 'Sowing': return <Sprout size={16} />;
      case 'Irrigation': return <Droplets size={16} />;
      case 'Harvest': return <Tractor size={16} />;
      default: return <Activity size={16} />;
    }
  };

  return (
    <>
      <div className="diary-container">
        <div className="diary-header">
          <h2><Book size={28} /> Digital Farm Diary</h2>
          <div className="diary-header-actions">
            <button onClick={() => setShowForm(!showForm)} className="diary-btn primary">
              {showForm ? <X size={18} /> : <Plus size={18} />}
              {showForm ? 'Cancel' : 'Add Entry'}
            </button>
            <button onClick={generatePDF} className="diary-btn secondary">
              <Download size={18} /> Export PDF
            </button>
            <button onClick={onClose} className="diary-btn close-modal-btn" aria-label="Close Diary" title="Close Diary">
              <X size={20} />
            </button>
          </div>
        </div>

        {upcomingReminders.length > 0 && (
          <div className="reminder-banner">
            <Clock size={16} />
            <span>You have <strong>{upcomingReminders.length}</strong> upcoming task{upcomingReminders.length > 1 ? 's' : ''} due soon!</span>
          </div>
        )}

        {showForm && (
          <form onSubmit={handleSubmit} className="diary-form">
            <div className="form-row">
              <div className="form-group">
                <label><Calendar size={14} /> Date *</label>
                <input type="date" name="date" value={formData.date} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label>Activity Type *</label>
                <select name="activityType" value={formData.activityType} onChange={handleInputChange}>
                  {ACTIVITY_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
            </div>
            <div className="form-group">
              <label>Notes *</label>
              <textarea name="notes" value={formData.notes} onChange={handleInputChange} rows={3} placeholder="Describe your farm activity..." required />
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Cost (₹)</label>
                <input type="number" name="cost" value={formData.cost} onChange={handleInputChange} placeholder="0.00" min="0" step="0.01" />
              </div>
              <div className="form-group">
                <label><Clock size={14} /> Set Reminder</label>
                <input type="date" name="reminderDate" value={formData.reminderDate} onChange={handleInputChange} />
              </div>
            </div>
            <div className="form-footer">
              <label className="checkbox-label">
                <input type="checkbox" name="isCompleted" checked={formData.isCompleted} onChange={handleInputChange}
                  style={{ width: '18px', height: '18px', accentColor: '#2ecc71' }} />
                Mark as Completed
              </label>
              <button type="button" onClick={resetForm} className="diary-btn secondary">Cancel</button>
              <button type="submit" className="diary-btn primary">
                {editingId ? 'Update Entry' : 'Save Entry'}
              </button>
            </div>
          </form>
        )}

        {entries.length === 0 && !showForm ? (
          <div className="empty-state">
            <Book className="icon" />
            <h3>No records found</h3>
            <p>Start logging your daily farm activities to keep track of your progress.</p>
          </div>
        ) : (
          <div className="diary-timeline-container">
            {entries.map((entry) => {
              const isUpcoming = !entry.isCompleted && entry.reminderDate && entry.reminderDate >= today;
              return (
                <div key={entry.id} className={`timeline-entry ${isUpcoming ? 'reminder' : ''}`}>
                  <div className="diary-timeline-dot"></div>
                  <div className="timeline-content">
                    <div className="timeline-header">
                      <span className={`entry-type-badge ${entry.activityType}`}>
                        {getActivityIcon(entry.activityType)}
                        {entry.activityType}
                      </span>
                      <div className="entry-date">
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <Calendar size={14} /> {new Date(entry.date).toLocaleDateString()}
                        </span>
                        {entry.reminderDate && (
                          <span className="reminder-tag">
                            <Clock size={12} /> Due: {new Date(entry.reminderDate).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="timeline-notes">{entry.notes}</p>
                    <div className="timeline-footer">
                      <div>{entry.cost && <span className="entry-cost">₹{entry.cost}</span>}</div>
                      <div className="entry-actions">
                        <button onClick={() => toggleStatus(entry.id)} className="entry-action-btn"
                          title={entry.isCompleted ? "Mark Pending" : "Mark Completed"}
                          style={{ color: entry.isCompleted ? '#2ecc71' : '#a0a0a0' }}>
                          <Check size={18} />
                        </button>
                        <button onClick={() => handleEdit(entry)} className="entry-action-btn" title="Edit">
                          <Edit2 size={18} />
                        </button>
                        <button onClick={() => handleDelete(entry.id)} className="entry-action-btn delete" title="Delete">
                          <Trash2 size={18} />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
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
    </>
  );
}
