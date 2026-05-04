import React, { useState, useEffect, useMemo } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { toast } from 'react-toastify';
import {
  Book, Plus, Download, Trash2, Edit2,
  Calendar, Clock, Check, X,
  Droplets, Sprout, Tractor, Activity, MessageCircle
} from 'lucide-react';
import './FarmDiary.css';
import SoilChatbot from './SoilChatbot';

const ACTIVITY_TYPES = ['Sowing', 'Irrigation', 'Fertilizer', 'Harvest', 'Pesticide', 'Other'];

const todayStr = () => new Date().toISOString().split('T')[0];

export default function FarmDiary({ onClose }) {
  const [entries, setEntries] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [showAdvisor, setShowAdvisor] = useState(false);

  const [formData, setFormData] = useState({
    date: todayStr(),
    activityType: 'Sowing',
    notes: '',
    cost: '',
    reminderDate: '',
    isCompleted: true
  });

  // Load once
  useEffect(() => {
    try {
      const saved = localStorage.getItem('fasalSaathiDiary');
      if (saved) setEntries(JSON.parse(saved));
    } catch (err) {
      console.error('Load error:', err);
    }
  }, []);

  // Save whenever entries change
  useEffect(() => {
    localStorage.setItem('fasalSaathiDiary', JSON.stringify(entries));
  }, [entries]);

  // Memoized sorted entries
  const sortedEntries = useMemo(() => {
    return [...entries].sort((a, b) => new Date(b.date) - new Date(a.date));
  }, [entries]);

  const handleInputChange = ({ target }) => {
    const { name, value, type, checked } = target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const resetForm = () => {
    setFormData({
      date: todayStr(),
      activityType: 'Sowing',
      notes: '',
      cost: '',
      reminderDate: '',
      isCompleted: true
    });
    setEditingId(null);
    setShowForm(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!formData.date || !formData.notes || !formData.activityType) {
      return toast.error('Please fill required fields');
    }

    setEntries(prev => {
      if (editingId) {
        toast.success('Entry updated');
        return prev.map(e =>
          e.id === editingId ? { ...formData, id: editingId } : e
        );
      }

      const newEntry = {
        ...formData,
        id: Date.now().toString()
      };

      toast.success('Entry added');
      return [newEntry, ...prev];
    });

    resetForm();
  };

  const handleDelete = (id) => {
    if (!window.confirm('Delete this entry?')) return;
    setEntries(prev => prev.filter(e => e.id !== id));
    toast.success('Deleted');
  };

  const toggleStatus = (id) => {
    setEntries(prev =>
      prev.map(e => {
        if (e.id !== id) return e;
        const updated = { ...e, isCompleted: !e.isCompleted };
        toast.info(updated.isCompleted ? 'Completed' : 'Marked pending');
        return updated;
      })
    );
  };

  const getIcon = (type) => {
    switch (type) {
      case 'Sowing': return <Sprout size={16} />;
      case 'Irrigation': return <Droplets size={16} />;
      case 'Harvest': return <Tractor size={16} />;
      default: return <Activity size={16} />;
    }
  };

  const generatePDF = () => {
    if (!entries.length) return toast.warning('No data');

    const doc = new jsPDF();

    const totalCost = entries.reduce(
      (sum, e) => sum + (parseFloat(e.cost) || 0),
      0
    );

    doc.setFontSize(18);
    doc.text('Farm Diary Report', 14, 20);

    doc.setFontSize(11);
    doc.text(`Entries: ${entries.length}`, 14, 30);
    doc.text(`Total Cost: ₹${totalCost.toFixed(2)}`, 14, 36);

    const rows = entries.map(e => [
      e.date,
      e.activityType,
      e.isCompleted ? 'Done' : 'Pending',
      e.notes,
      e.cost || '-',
      e.reminderDate || '-'
    ]);

    autoTable(doc, {
      head: [['Date', 'Activity', 'Status', 'Notes', 'Cost', 'Reminder']],
      body: rows,
      startY: 45
    });

    doc.save(`farm-diary-${todayStr()}.pdf`);
    toast.success('PDF generated');
  };

  return (
    <div className="diary-container">

      {/* Header */}
      <div className="diary-header">
        <h2><Book size={26} /> Farm Diary</h2>

        <div className="diary-header-actions">
          <button onClick={() => setShowForm(s => !s)} className="diary-btn primary">
            {showForm ? <X size={16} /> : <Plus size={16} />}
            {showForm ? 'Cancel' : 'Add'}
          </button>

          <button onClick={generatePDF} className="diary-btn secondary">
            <Download size={16} /> Export
          </button>

          <button onClick={onClose} className="diary-btn close-modal-btn">
            <X size={18} />
          </button>
        </div>
      </div>

      {/* Form */}
      {showForm && (
        <form onSubmit={handleSubmit} className="diary-form">
          <input type="date" name="date" value={formData.date} onChange={handleInputChange} />
          
          <select name="activityType" value={formData.activityType} onChange={handleInputChange}>
            {ACTIVITY_TYPES.map(t => <option key={t}>{t}</option>)}
          </select>

          <textarea
            name="notes"
            value={formData.notes}
            onChange={handleInputChange}
            placeholder="Notes..."
          />

          <input type="number" name="cost" value={formData.cost} onChange={handleInputChange} />

          <input type="date" name="reminderDate" value={formData.reminderDate} onChange={handleInputChange} />

          <label>
            <input
              type="checkbox"
              name="isCompleted"
              checked={formData.isCompleted}
              onChange={handleInputChange}
            />
            Completed
          </label>

          <button type="submit" className="diary-btn primary">
            {editingId ? 'Update' : 'Save'}
          </button>
        </form>
      )}

      {/* List */}
      {sortedEntries.length === 0 ? (
        <div className="empty-state">
          <Book />
          <p>No entries yet</p>
        </div>
      ) : (
        <div className="diary-timeline-container">
          {sortedEntries.map(entry => {
            const isUpcoming =
              !entry.isCompleted &&
              entry.reminderDate &&
              new Date(entry.reminderDate) >= new Date(todayStr());

            return (
              <div key={entry.id} className={`timeline-entry ${isUpcoming ? 'reminder' : ''}`}>
                <div className="timeline-content">

                  <div className="timeline-header">
                    <span className="entry-type-badge">
                      {getIcon(entry.activityType)}
                      {entry.activityType}
                    </span>

                    <span>
                      <Calendar size={14} />
                      {new Date(entry.date).toLocaleDateString()}
                    </span>
                  </div>

                  <p>{entry.notes}</p>

                  <div className="timeline-footer">
                    <span>₹{entry.cost}</span>

                    <div className="entry-actions">
                      <button onClick={() => toggleStatus(entry.id)}>
                        <Check size={16} />
                      </button>
                      <button onClick={() => {
                        setFormData(entry);
                        setEditingId(entry.id);
                        setShowForm(true);
                      }}>
                        <Edit2 size={16} />
                      </button>
                      <button onClick={() => handleDelete(entry.id)}>
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>

                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* AI Advisor */}
      <button className="advisor-fab" onClick={() => setShowAdvisor(true)}>
        <MessageCircle />
      </button>

      {showAdvisor && (
        <div className="advisor-overlay" onClick={() => setShowAdvisor(false)}>
          <div onClick={e => e.stopPropagation()}>
            <SoilChatbot onClose={() => setShowAdvisor(false)} />
          </div>
        </div>
      )}

    </div>
  );
}