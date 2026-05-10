import React, { useState, useEffect, useMemo } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { toast } from 'react-toastify';
import { 
  FaFileInvoiceDollar, 
  FaPlus, 
  FaDownload, 
  FaTrashAlt, 
  FaEdit, 
  FaChartBar, 
  FaArrowLeft,
  FaMoneyBillWave,
  FaReceipt,
  FaArrowUp,
  FaArrowDown
} from 'react-icons/fa';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell
} from 'recharts';
import './FarmFinance.css';

export default function FarmFinance() {
  const [incomeEntries, setIncomeEntries] = useState([]);
  const [expenseEntries, setExpenseEntries] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  
  const [formData, setFormData] = useState({
    date: new Date().toISOString().split('T')[0],
    cropName: '',
    quantity: '',
    pricePerUnit: '',
    notes: ''
  });

  // Load data from localStorage
  useEffect(() => {
    const savedIncome = localStorage.getItem('fasalSaathiIncome');
    if (savedIncome) {
      try {
        setIncomeEntries(JSON.parse(savedIncome));
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error('Error loading income data', e);
      }
    }

    const savedExpenses = localStorage.getItem('fasalSaathiDiary');
    if (savedExpenses) {
      try {
        setExpenseEntries(JSON.parse(savedExpenses));
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error('Error loading expense data', e);
      }
    }
  }, []);

  // Save income to localStorage
  useEffect(() => {
    localStorage.setItem('fasalSaathiIncome', JSON.stringify(incomeEntries));
  }, [incomeEntries]);

  // Calculations
  const totalIncome = useMemo(() => {
    return incomeEntries.reduce((sum, entry) => sum + (parseFloat(entry.quantity) * parseFloat(entry.pricePerUnit) || 0), 0);
  }, [incomeEntries]);

  const totalExpense = useMemo(() => {
    return expenseEntries.reduce((sum, entry) => sum + (parseFloat(entry.cost) || 0), 0);
  }, [expenseEntries]);

  const netProfit = totalIncome - totalExpense;

  const chartData = useMemo(() => {
    return [
      { name: 'Income', amount: totalIncome, fill: '#2ecc71' },
      { name: 'Expenses', amount: totalExpense, fill: '#e74c3c' }
    ];
  }, [totalIncome, totalExpense]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.cropName || !formData.quantity || !formData.pricePerUnit) {
      toast.error('Please fill in required fields');
      return;
    }

    if (editingId) {
      setIncomeEntries(incomeEntries.map(entry => 
        entry.id === editingId ? { ...formData, id: editingId } : entry
      ));
      toast.success('Income entry updated!');
    } else {
      const newEntry = {
        ...formData,
        id: Date.now().toString()
      };
      setIncomeEntries([newEntry, ...incomeEntries]);
      toast.success('Income entry added!');
    }

    resetForm();
  };

  const handleEdit = (entry) => {
    setFormData(entry);
    setEditingId(entry.id);
    setShowForm(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleDelete = (id) => {
    if (window.confirm('Delete this income record?')) {
      setIncomeEntries(incomeEntries.filter(e => e.id !== id));
      toast.info('Income record removed');
    }
  };

  const resetForm = () => {
    setFormData({
      date: new Date().toISOString().split('T')[0],
      cropName: '',
      quantity: '',
      pricePerUnit: '',
      notes: ''
    });
    setEditingId(null);
    setShowForm(false);
  };

  const generatePDF = () => {
    const doc = new jsPDF();
    
    // Header
    doc.setFontSize(22);
    doc.setTextColor(46, 204, 113);
    doc.text('Fasal Saathi - Financial Report', 14, 20);
    
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Report Date: ${new Date().toLocaleDateString()}`, 14, 28);
    
    // Summary
    doc.setFontSize(14);
    doc.setTextColor(0);
    doc.text('Financial Summary', 14, 40);
    
    autoTable(doc, {
      startY: 45,
      head: [['Metric', 'Amount (INR)']],
      body: [
        ['Total Revenue', `Rs. ${totalIncome.toFixed(2)}`],
        ['Total Operational Expenses', `Rs. ${totalExpense.toFixed(2)}`],
        ['Net Profit/Loss', `Rs. ${netProfit.toFixed(2)}`]
      ],
      theme: 'striped',
      headStyles: { fillColor: [46, 204, 113] }
    });

    // Income Table
    doc.text('Revenue Breakdown', 14, doc.lastAutoTable.finalY + 15);
    const incomeRows = incomeEntries.map(e => [
      e.date,
      e.cropName,
      e.quantity,
      `Rs. ${e.pricePerUnit}`,
      `Rs. ${(e.quantity * e.pricePerUnit).toFixed(2)}`
    ]);

    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 20,
      head: [['Date', 'Crop', 'Quantity', 'Price/Unit', 'Total']],
      body: incomeRows,
      theme: 'grid'
    });

    // Expense Table (Summary)
    doc.text('Expense Summary (from Logs)', 14, doc.lastAutoTable.finalY + 15);
    const expenseRows = expenseEntries.filter(e => e.cost).map(e => [
      e.date,
      e.activityType,
      e.notes.substring(0, 30) + '...',
      `Rs. ${e.cost}`
    ]);

    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 20,
      head: [['Date', 'Activity', 'Details', 'Cost']],
      body: expenseRows,
      theme: 'grid'
    });

    doc.save(`Farm_Finance_Report_${new Date().toISOString().split('T')[0]}.pdf`);
    toast.success('Financial Report Exported!');
  };

  return (
    <div className="finance-container">
      <div className="finance-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Link to="/advisor" className="finance-btn secondary" style={{ padding: '0.5rem' }}>
            <FaArrowLeft />
          </Link>
          <h2><FaFileInvoiceDollar /> Farm Finance</h2>
        </div>
        <div className="finance-actions">
          <button onClick={() => setShowForm(!showForm)} className="finance-btn primary">
            <FaPlus /> {showForm ? 'Cancel' : 'Record Income'}
          </button>
          <button onClick={generatePDF} className="finance-btn secondary">
            <FaDownload /> Export Report
          </button>
        </div>
      </div>

      <div className="finance-summary">
        <div className="summary-card income">
          <span className="label">Total Revenue</span>
          <span className="value">₹{totalIncome.toLocaleString()}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#2ecc71', fontSize: '0.8rem' }}>
            <FaArrowUp /> Active Season
          </div>
        </div>
        <div className="summary-card expense">
          <span className="label">Operational Costs</span>
          <span className="value">₹{totalExpense.toLocaleString()}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#e74c3c', fontSize: '0.8rem' }}>
            <FaArrowDown /> From Logs
          </div>
        </div>
        <div className="summary-card profit">
          <span className="label">Net Profit</span>
          <span className="value" style={{ color: netProfit >= 0 ? '#3498db' : '#e74c3c' }}>
            ₹{netProfit.toLocaleString()}
          </span>
          <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
            Profitability Ratio: {totalIncome > 0 ? ((netProfit / totalIncome) * 100).toFixed(1) : 0}%
          </div>
        </div>
      </div>

      <div className="finance-charts">
        <div className="chart-header">
          <h3><FaChartBar /> Financial Overview</h3>
        </div>
        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#a0a0a0' }} />
              <YAxis axisLine={false} tickLine={false} tick={{ fill: '#a0a0a0' }} />
              <Tooltip 
                contentStyle={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Bar dataKey="amount" radius={[8, 8, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="finance-content-grid">
        <div className="finance-section">
          <h3 className="section-title"><FaMoneyBillWave /> {editingId ? 'Edit Record' : 'Log New Revenue'}</h3>
          <form onSubmit={handleSubmit} className="finance-form">
            <div className="form-group">
              <label>Sale Date</label>
              <input type="date" name="date" value={formData.date} onChange={handleInputChange} className="finance-input" required />
            </div>
            <div className="form-group">
              <label>Crop / Produce Sold</label>
              <input type="text" name="cropName" value={formData.cropName} onChange={handleInputChange} className="finance-input" placeholder="e.g. Basmati Rice" required />
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Quantity (Quintals)</label>
                <input type="number" name="quantity" value={formData.quantity} onChange={handleInputChange} className="finance-input" placeholder="0.00" required />
              </div>
              <div className="form-group">
                <label>Price (per Quintal)</label>
                <input type="number" name="pricePerUnit" value={formData.pricePerUnit} onChange={handleInputChange} className="finance-input" placeholder="₹0.00" required />
              </div>
            </div>
            <div className="form-group">
              <label>Notes (Optional)</label>
              <textarea name="notes" value={formData.notes} onChange={handleInputChange} className="finance-input" placeholder="Market name, commission details..." rows="2" />
            </div>
            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              {editingId && <button type="button" onClick={resetForm} className="finance-btn secondary" style={{ flex: 1 }}>Cancel</button>}
              <button type="submit" className="finance-btn primary" style={{ flex: 2 }}>
                {editingId ? 'Update Record' : 'Save Income'}
              </button>
            </div>
          </form>
        </div>

        <div className="finance-section">
          <h3 className="section-title"><FaReceipt /> Income History</h3>
          <div className="data-table-container">
            {incomeEntries.length === 0 ? (
              <div className="empty-state">
                <FaReceipt className="icon" />
                <p>No income records yet.<br/>Start by logging your crop sales.</p>
              </div>
            ) : (
              <table className="finance-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Crop</th>
                    <th>Total</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {incomeEntries.map(entry => (
                    <tr key={entry.id}>
                      <td>{entry.date}</td>
                      <td>
                        <div style={{ fontWeight: '600' }}>{entry.cropName}</div>
                        <div style={{ fontSize: '0.75rem', color: '#a0a0a0' }}>{entry.quantity} Q @ ₹{entry.pricePerUnit}</div>
                      </td>
                      <td style={{ color: '#2ecc71', fontWeight: '700' }}>
                        ₹{(entry.quantity * entry.pricePerUnit).toLocaleString()}
                      </td>
                      <td>
                        <div className="action-btns">
                          <button onClick={() => handleEdit(entry)} className="action-btn" title="Edit"><FaEdit /></button>
                          <button onClick={() => handleDelete(entry.id)} className="action-btn delete" title="Delete"><FaTrashAlt /></button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
