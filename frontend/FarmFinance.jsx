import React, { useEffect, useMemo, useState } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { toast } from 'react-toastify';
import {
  FaArrowDown,
  FaArrowLeft,
  FaArrowUp,
  FaCalculator,
  FaChartBar,
  FaChartLine,
  FaCheckCircle,
  FaClipboardList,
  FaDownload,
  FaEdit,
  FaExclamationTriangle,
  FaFileInvoiceDollar,
  FaMoneyBillWave,
  FaPlus,
  FaReceipt,
  FaSeedling,
  FaTrashAlt,
  FaUniversity,
  FaWallet,
} from 'react-icons/fa';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Bar,
  Cell,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './FarmFinance.css';
import { loadVersionedArray, saveVersionedArray } from './utils/versionedStorage';
import apiClient from './lib/apiClient';

const INCOME_STORAGE_KEY = 'fasalSaathiIncome';
const EXPENSE_STORAGE_KEY = 'fasalSaathiDiary';
const FINANCE_STORAGE_VERSION = 1;
const MAX_FINANCE_ENTRIES = 250;

export default function FarmFinance() {
  const [incomeEntries, setIncomeEntries] = useState([]);
  const [expenseEntries, setExpenseEntries] = useState([]);
  const [expenseTrackerData, setExpenseTrackerData] = useState({
    seedCost: '',
    fertilizerCost: '',
    irrigationCost: '',
    laborCost: '',
  });
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [loanFormTouched, setLoanFormTouched] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loanApplications, setLoanApplications] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [financeNotice, setFinanceNotice] = useState('');

  const [formData, setFormData] = useState({
    date: new Date().toISOString().split('T')[0],
    cropName: '',
    quantity: '',
    pricePerUnit: '',
    notes: '',
  });

  const [loanFormData, setLoanFormData] = useState({
    farmerName: '',
    cropType: 'Rice',
    acreage: '',
    annualRevenue: '',
    annualOperatingCost: '',
    existingDebt: '',
    emergencyFund: '',
    creditScore: '700',
    requestedLoanAmount: '',
    loanTenureMonths: '36',
    irrigationCost: '',
    laborCost: '',
    farmLocation: '',
    selectedLender: '',
    notes: '',
  });

  const currencyFormatter = useMemo(
    () => new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }),
    []
  );

  const formatCurrency = (value) => currencyFormatter.format(Number(value) || 0);

  useEffect(() => {
    setIncomeEntries(loadVersionedArray(INCOME_STORAGE_KEY, {
      version: FINANCE_STORAGE_VERSION,
      fallback: [],
      maxItems: MAX_FINANCE_ENTRIES,
    }));

    setExpenseEntries(loadVersionedArray(EXPENSE_STORAGE_KEY, {
      version: FINANCE_STORAGE_VERSION,
      fallback: [],
      maxItems: MAX_FINANCE_ENTRIES,
    }));
  }, []);

  useEffect(() => {
    const saved = saveVersionedArray(INCOME_STORAGE_KEY, incomeEntries, {
      version: FINANCE_STORAGE_VERSION,
      maxItems: MAX_FINANCE_ENTRIES,
    });

    if (!saved) {
      setFinanceNotice('Income history is temporarily full. Older records were retained in memory only.');
    }
  }, [incomeEntries]);

  useEffect(() => {
    saveVersionedArray(EXPENSE_STORAGE_KEY, expenseEntries, {
      version: FINANCE_STORAGE_VERSION,
      maxItems: MAX_FINANCE_ENTRIES,
    });
  }, [expenseEntries]);

  const totalIncome = useMemo(() => {
    return incomeEntries.reduce(
      (sum, entry) => sum + (parseFloat(entry.quantity) * parseFloat(entry.pricePerUnit) || 0),
      0
    );
  }, [incomeEntries]);

  const totalExpense = useMemo(() => {
    return expenseEntries.reduce((sum, entry) => sum + (parseFloat(entry.cost) || 0), 0);
  }, [expenseEntries]);

  const estimatedTrackerCost = useMemo(() => {
    return Object.values(expenseTrackerData).reduce(
      (sum, value) => sum + (parseFloat(value) || 0),
      0
    );
  }, [expenseTrackerData]);

  const netProfit = totalIncome - totalExpense;

  const chartData = useMemo(() => ([
    { name: 'Income', amount: totalIncome, fill: '#2ecc71' },
    { name: 'Expenses', amount: totalExpense, fill: '#e74c3c' },
  ]), [totalIncome, totalExpense]);

  useEffect(() => {
    if (loanFormTouched) return;

    const estimatedRevenue = Math.max(Math.round(totalIncome), 0);
    const estimatedCost = Math.max(Math.round(totalExpense), 0);

    setLoanFormData(prev => ({
      ...prev,
      cropType: prev.cropType || incomeEntries[0]?.cropName || 'Rice',
      annualRevenue: estimatedRevenue ? String(estimatedRevenue) : prev.annualRevenue,
      annualOperatingCost: estimatedCost ? String(estimatedCost) : prev.annualOperatingCost,
      requestedLoanAmount: estimatedRevenue ? String(Math.max(Math.round(estimatedRevenue * 0.3), 50000)) : prev.requestedLoanAmount,
    }));
  }, [incomeEntries, totalExpense, totalIncome, loanFormTouched]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleLoanInputChange = (e) => {
    const { name, value } = e.target;
    setLoanFormTouched(true);
    setLoanFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleTrackerInputChange = (e) => {
    const { name, value } = e.target;
    setExpenseTrackerData(prev => ({ ...prev, [name]: value }));
  };

  const applyTrackerToLoanForm = () => {
    setLoanFormTouched(true);
    setLoanFormData(prev => ({
      ...prev,
      annualOperatingCost: String(Math.max(Math.round(estimatedTrackerCost), 0)),
      irrigationCost: expenseTrackerData.irrigationCost || prev.irrigationCost,
      laborCost: expenseTrackerData.laborCost || prev.laborCost,
      notes: prev.notes || 'Auto-filled from expense tracker',
    }));
    toast.success('Estimated cost copied to loan planner');
  };

  const resetExpenseTracker = () => {
    setExpenseTrackerData({
      seedCost: '',
      fertilizerCost: '',
      irrigationCost: '',
      laborCost: '',
    });
  };

  const populateFromFarmTotals = () => {
    setLoanFormTouched(true);
    setLoanFormData(prev => ({
      ...prev,
      cropType: incomeEntries[0]?.cropName || prev.cropType || 'Rice',
      annualRevenue: String(Math.max(Math.round(totalIncome), 0)),
      annualOperatingCost: String(Math.max(Math.round(totalExpense), 0)),
      requestedLoanAmount: String(Math.max(Math.round(Math.max(totalIncome - totalExpense, 0) * 0.35), 50000)),
      notes: prev.notes || 'Auto-filled from current farm records',
    }));
  };

  const buildFinancePayload = () => ({
    farmer_name: loanFormData.farmerName || 'Farm Operator',
    crop_type: loanFormData.cropType,
    acreage: Number(loanFormData.acreage) || 0,
    annual_revenue: Number(loanFormData.annualRevenue) || 0,
    annual_operating_cost: Number(loanFormData.annualOperatingCost) || 0,
    existing_debt: Number(loanFormData.existingDebt) || 0,
    emergency_fund: Number(loanFormData.emergencyFund) || 0,
    credit_score: Number(loanFormData.creditScore) || 650,
    requested_loan_amount: Number(loanFormData.requestedLoanAmount) || 0,
    loan_tenure_months: Number(loanFormData.loanTenureMonths) || 36,
    irrigation_cost: Number(loanFormData.irrigationCost) || 0,
    labor_cost: Number(loanFormData.laborCost) || 0,
    selected_lender: loanFormData.selectedLender || '',
    farm_location: loanFormData.farmLocation || '',
    notes: loanFormData.notes || '',
  });

  const postFinanceRequest = async (path, payload) => {
    // Use apiClient so the Firebase auth token is automatically injected via
    // the Axios request interceptor. Raw fetch() has no Authorization header,
    // causing every request to be rejected with 401/403 by the backend's
    // rbac_manager.raise_if_unauthorized() check.
    const response = await apiClient.post(path, payload);
    const responseData = response.data;
    if (responseData?.success === false) {
      throw new Error(responseData?.detail || responseData?.message || 'Finance request failed');
    }
    return responseData.data;
  };

  const handleAnalyzeFinance = async () => {
    try {
      setIsAnalyzing(true);
      setFinanceNotice('');
      const analysis = await postFinanceRequest('/api/farm-finance/analyze', buildFinancePayload());
      setAnalysisResult(analysis);
      toast.success('Financial assessment generated');
    } catch (error) {
      setFinanceNotice(error.message);
      toast.error(error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCreateApplication = async () => {
    try {
      setIsApplying(true);
      setFinanceNotice('');
      const application = await postFinanceRequest('/api/farm-finance/applications', buildFinancePayload());
      setLoanApplications(prev => [application, ...prev]);
      toast.success(`Loan application created: ${application.application_id}`);
    } catch (error) {
      setFinanceNotice(error.message);
      toast.error(error.message);
    } finally {
      setIsApplying(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.cropName || !formData.quantity || !formData.pricePerUnit) {
      toast.error('Please fill in required fields');
      return;
    }

    if (editingId) {
      setIncomeEntries(incomeEntries.map(entry => (
        entry.id === editingId ? { ...formData, id: editingId } : entry
      )));
      toast.success('Income entry updated!');
    } else {
      const newEntry = {
        ...formData,
        id: Date.now().toString(),
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
      setIncomeEntries(incomeEntries.filter(entry => entry.id !== id));
      toast.info('Income record removed');
    }
  };

  const resetForm = () => {
    setFormData({
      date: new Date().toISOString().split('T')[0],
      cropName: '',
      quantity: '',
      pricePerUnit: '',
      notes: '',
    });
    setEditingId(null);
    setShowForm(false);
  };

  const generatePDF = () => {
    const doc = new jsPDF();

    doc.setFontSize(22);
    doc.setTextColor(46, 204, 113);
    doc.text('Fasal Saathi - Financial Report', 14, 20);

    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Report Date: ${new Date().toLocaleDateString()}`, 14, 28);

    doc.setFontSize(14);
    doc.setTextColor(0);
    doc.text('Financial Summary', 14, 40);

    autoTable(doc, {
      startY: 45,
      head: [['Metric', 'Amount (INR)']],
      body: [
        ['Total Revenue', `Rs. ${totalIncome.toFixed(2)}`],
        ['Total Operational Expenses', `Rs. ${totalExpense.toFixed(2)}`],
        ['Net Profit/Loss', `Rs. ${netProfit.toFixed(2)}`],
      ],
      theme: 'striped',
      headStyles: { fillColor: [46, 204, 113] },
    });

    doc.text('Revenue Breakdown', 14, doc.lastAutoTable.finalY + 15);
    const incomeRows = incomeEntries.map(entry => [
      entry.date,
      entry.cropName,
      entry.quantity,
      `Rs. ${entry.pricePerUnit}`,
      `Rs. ${(entry.quantity * entry.pricePerUnit).toFixed(2)}`,
    ]);

    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 20,
      head: [['Date', 'Crop', 'Quantity', 'Price/Unit', 'Total']],
      body: incomeRows,
      theme: 'grid',
    });

    doc.text('Expense Summary (from Logs)', 14, doc.lastAutoTable.finalY + 15);
    const expenseRows = expenseEntries.filter(entry => entry.cost).map(entry => [
      entry.date,
      entry.activityType,
      entry.notes.substring(0, 30) + '...',
      `Rs. ${entry.cost}`,
    ]);

    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 20,
      head: [['Date', 'Activity', 'Details', 'Cost']],
      body: expenseRows,
      theme: 'grid',
    });

    doc.save(`Farm_Finance_Report_${new Date().toISOString().split('T')[0]}.pdf`);
    toast.success('Financial Report Exported!');
  };

  const loanHealthLabel = analysisResult?.risk_level || (netProfit >= 0 ? 'Stable' : 'At Risk');

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
          <span className="value">{formatCurrency(totalIncome)}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#2ecc71', fontSize: '0.8rem' }}>
            <FaArrowUp /> Active Season
          </div>
        </div>
        <div className="summary-card expense">
          <span className="label">Operational Costs</span>
          <span className="value">{formatCurrency(totalExpense)}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#e74c3c', fontSize: '0.8rem' }}>
            <FaArrowDown /> From Logs
          </div>
        </div>
        <div className="summary-card profit">
          <span className="label">Net Profit</span>
          <span className="value" style={{ color: netProfit >= 0 ? '#3498db' : '#e74c3c' }}>
            {formatCurrency(netProfit)}
          </span>
          <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
            Profitability Ratio: {totalIncome > 0 ? ((netProfit / totalIncome) * 100).toFixed(1) : 0}%
          </div>
        </div>
        <div className="summary-card loan-health">
          <span className="label">Loan Health</span>
          <span className="value">{analysisResult ? `${analysisResult.financial_health_score}/100` : 'Ready'}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: analysisResult?.risk_level === 'Low' ? '#2ecc71' : '#f39c12', fontSize: '0.8rem' }}>
            {analysisResult?.risk_level === 'Low' ? <FaCheckCircle /> : <FaExclamationTriangle />}
            {loanHealthLabel}
          </div>
        </div>
        <div className="summary-card projected-cost">
          <span className="label">Estimated Crop Cost</span>
          <span className="value">{formatCurrency(estimatedTrackerCost)}</span>
          <div style={{ fontSize: '0.8rem', opacity: 0.85 }}>
            Seed, fertilizer, irrigation, and labor
          </div>
        </div>
      </div>

      <div className="finance-section expense-tracker-panel">
        <div className="expense-tracker-header">
          <div>
            <h3 className="section-title"><FaSeedling /> Farm Expense Tracker</h3>
            <p>Enter the main seasonal costs to estimate your total crop spending before sales start coming in.</p>
          </div>
          <div className="expense-tracker-total">
            <span>Estimated Total</span>
            <strong>{formatCurrency(estimatedTrackerCost)}</strong>
          </div>
        </div>

        <div className="finance-ai-grid expense-tracker-grid">
          <div className="finance-section expense-tracker-form">
            <div className="form-row">
              <div className="form-group">
                <label>Seed Costs</label>
                <input type="number" name="seedCost" value={expenseTrackerData.seedCost} onChange={handleTrackerInputChange} className="finance-input" placeholder="0" min="0" />
              </div>
              <div className="form-group">
                <label>Fertilizer Expenses</label>
                <input type="number" name="fertilizerCost" value={expenseTrackerData.fertilizerCost} onChange={handleTrackerInputChange} className="finance-input" placeholder="0" min="0" />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Irrigation Costs</label>
                <input type="number" name="irrigationCost" value={expenseTrackerData.irrigationCost} onChange={handleTrackerInputChange} className="finance-input" placeholder="0" min="0" />
              </div>
              <div className="form-group">
                <label>Labor Expenses</label>
                <input type="number" name="laborCost" value={expenseTrackerData.laborCost} onChange={handleTrackerInputChange} className="finance-input" placeholder="0" min="0" />
              </div>
            </div>
            <div className="expense-tracker-actions">
              <button type="button" className="finance-btn secondary" onClick={resetExpenseTracker}>
                Clear
              </button>
              <button type="button" className="finance-btn primary" onClick={applyTrackerToLoanForm}>
                Use in Loan Planner
              </button>
            </div>
          </div>

          <div className="finance-section expense-tracker-breakdown">
            <h3 className="section-title"><FaClipboardList /> Cost Breakdown</h3>
            <div className="tracker-breakdown-list">
              <div className="tracker-breakdown-item">
                <span>Seed</span>
                <strong>{formatCurrency(expenseTrackerData.seedCost)}</strong>
              </div>
              <div className="tracker-breakdown-item">
                <span>Fertilizer</span>
                <strong>{formatCurrency(expenseTrackerData.fertilizerCost)}</strong>
              </div>
              <div className="tracker-breakdown-item">
                <span>Irrigation</span>
                <strong>{formatCurrency(expenseTrackerData.irrigationCost)}</strong>
              </div>
              <div className="tracker-breakdown-item">
                <span>Labor</span>
                <strong>{formatCurrency(expenseTrackerData.laborCost)}</strong>
              </div>
            </div>
            <div className="tracker-breakdown-total">
              <span>Estimated Total Cost</span>
              <strong>{formatCurrency(estimatedTrackerCost)}</strong>
            </div>
          </div>
        </div>
      </div>

      <div className="finance-ai-panel">
        <div className="finance-ai-header">
          <div>
            <h3><FaCalculator /> AI Loan Planner</h3>
            <p>Run a financial assessment, match lenders, and create a loan application from the same farm record.</p>
          </div>
          <div className="finance-ai-actions">
            <button type="button" className="finance-btn secondary" onClick={populateFromFarmTotals}>
              <FaSeedling /> Use Farm Totals
            </button>
            <button type="button" className="finance-btn primary" onClick={handleAnalyzeFinance} disabled={isAnalyzing}>
              <FaChartLine /> {isAnalyzing ? 'Analyzing...' : 'Assess Finance'}
            </button>
            <button type="button" className="finance-btn secondary" onClick={handleCreateApplication} disabled={isApplying}>
              <FaUniversity /> {isApplying ? 'Submitting...' : 'Create Loan Application'}
            </button>
          </div>
        </div>

        {financeNotice && <div className="finance-notice">{financeNotice}</div>}

        <div className="finance-ai-grid">
          <form className="finance-section finance-ai-form" onSubmit={(event) => { event.preventDefault(); handleAnalyzeFinance(); }}>
            <h3 className="section-title"><FaWallet /> Financial Profile</h3>
            <div className="form-group">
              <label>Farmer Name</label>
              <input type="text" name="farmerName" value={loanFormData.farmerName} onChange={handleLoanInputChange} className="finance-input" placeholder="Farmer / Business name" />
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Crop Type</label>
                <input type="text" name="cropType" value={loanFormData.cropType} onChange={handleLoanInputChange} className="finance-input" placeholder="Rice, Wheat, Tomato" />
              </div>
              <div className="form-group">
                <label>Acreage</label>
                <input type="number" name="acreage" value={loanFormData.acreage} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Annual Revenue</label>
                <input type="number" name="annualRevenue" value={loanFormData.annualRevenue} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
              <div className="form-group">
                <label>Operating Cost</label>
                <input type="number" name="annualOperatingCost" value={loanFormData.annualOperatingCost} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Existing Debt</label>
                <input type="number" name="existingDebt" value={loanFormData.existingDebt} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
              <div className="form-group">
                <label>Emergency Fund</label>
                <input type="number" name="emergencyFund" value={loanFormData.emergencyFund} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Credit Score</label>
                <input type="number" name="creditScore" value={loanFormData.creditScore} onChange={handleLoanInputChange} className="finance-input" placeholder="700" />
              </div>
              <div className="form-group">
                <label>Requested Loan Amount</label>
                <input type="number" name="requestedLoanAmount" value={loanFormData.requestedLoanAmount} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Loan Tenure (Months)</label>
                <input type="number" name="loanTenureMonths" value={loanFormData.loanTenureMonths} onChange={handleLoanInputChange} className="finance-input" placeholder="36" />
              </div>
              <div className="form-group">
                <label>Selected Lender (Optional)</label>
                <input type="text" name="selectedLender" value={loanFormData.selectedLender} onChange={handleLoanInputChange} className="finance-input" placeholder="Regional Cooperative Bank" />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Irrigation Cost</label>
                <input type="number" name="irrigationCost" value={loanFormData.irrigationCost} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
              <div className="form-group">
                <label>Labor Cost</label>
                <input type="number" name="laborCost" value={loanFormData.laborCost} onChange={handleLoanInputChange} className="finance-input" placeholder="0" />
              </div>
            </div>
            <div className="form-group">
              <label>Farm Location</label>
              <input type="text" name="farmLocation" value={loanFormData.farmLocation} onChange={handleLoanInputChange} className="finance-input" placeholder="District / State" />
            </div>
            <div className="form-group">
              <label>Notes</label>
              <textarea name="notes" value={loanFormData.notes} onChange={handleLoanInputChange} className="finance-input" rows="2" placeholder="Add crop cycle, repayment, or compliance notes" />
            </div>
          </form>

          <div className="finance-section finance-ai-results">
            <h3 className="section-title"><FaClipboardList /> AI Recommendations</h3>
            {!analysisResult ? (
              <div className="empty-state compact">
                <FaCalculator className="icon" />
                <p>Run a finance assessment to see score, lender matches, and repayment guidance.</p>
              </div>
            ) : (
              <>
                <div className="result-summary-grid">
                  <div className="result-card highlight">
                    <span className="label">Health Score</span>
                    <strong>{analysisResult.financial_health_score}/100</strong>
                  </div>
                  <div className="result-card">
                    <span className="label">Risk Level</span>
                    <strong>{analysisResult.risk_level}</strong>
                  </div>
                  <div className="result-card">
                    <span className="label">Recommended Loan</span>
                    <strong>{formatCurrency(analysisResult.recommended_loan_amount)}</strong>
                  </div>
                  <div className="result-card">
                    <span className="label">Max Affordable EMI</span>
                    <strong>{formatCurrency(analysisResult.max_affordable_emi)}</strong>
                  </div>
                </div>

                <div className="metric-pill-row">
                  <span className="metric-pill">Profit Margin: {analysisResult.profit_margin_pct}%</span>
                  <span className="metric-pill">Debt Ratio: {analysisResult.debt_ratio_pct}%</span>
                  <span className="metric-pill">Emergency Cover: {analysisResult.emergency_cover_months} months</span>
                </div>

                <div className="recommendation-block">
                  <h4>Best Match</h4>
                  <p><strong>{analysisResult.selected_product?.product_name}</strong> from {analysisResult.selected_product?.lender_name}</p>
                  <p>{analysisResult.selected_product?.description}</p>
                </div>

                <div className="recommendation-block">
                  <h4>Suggested Lenders</h4>
                  <div className="lender-list">
                    {analysisResult.lender_matches?.slice(0, 3).map((lender) => (
                      <div key={lender.lender_name} className="lender-card">
                        <strong>{lender.lender_name}</strong>
                        <span>{lender.product_name}</span>
                        <small>Fit Score: {lender.fit_score}/100</small>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="recommendation-block">
                  <h4>Required Documents</h4>
                  <ul className="finance-list">
                    {analysisResult.required_documents?.map((item) => <li key={item}>{item}</li>)}
                  </ul>
                </div>

                <div className="recommendation-block">
                  <h4>Action Plan</h4>
                  <ul className="finance-list">
                    {analysisResult.action_plan?.map((item) => <li key={item}>{item}</li>)}
                  </ul>
                </div>
              </>
            )}
          </div>
        </div>

        {loanApplications.length > 0 && (
          <div className="finance-section application-panel">
            <h3 className="section-title"><FaUniversity /> Recent Loan Applications</h3>
            <div className="application-grid">
              {loanApplications.map((application) => (
                <div className="application-card" key={application.application_id}>
                  <div className="application-card-header">
                    <strong>{application.application_id}</strong>
                    <span className={`status-tag ${application.status}`}>{application.status.replace('_', ' ')}</span>
                  </div>
                  <p>{application.selected_lender}</p>
                  <small>{application.crop_type} • {formatCurrency(application.recommended_amount)}</small>
                  <small>Score: {application.assessment_score}/100</small>
                </div>
              ))}
            </div>
          </div>
        )}

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
              <Legend />
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
                <p>No income records yet.<br />Start by logging your crop sales.</p>
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
                        {formatCurrency(entry.quantity * entry.pricePerUnit)}
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
