import React from 'react';
import { FaExclamationTriangle, FaRedo } from 'react-icons/fa';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '3rem 1.5rem',
          textAlign: 'center',
          background: 'var(--bg-secondary, #f8fafc)',
          borderRadius: '16px',
          border: '1px solid var(--border-color, #e2e8f0)',
          margin: '2rem auto',
          maxWidth: '600px',
          boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.5rem'
        }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ef4444',
            fontSize: '2rem'
          }}>
            <FaExclamationTriangle />
          </div>
          
          <div style={{ gap: '0.5rem', display: 'flex', flexDirection: 'column' }}>
            <h2 style={{ 
              fontSize: '1.5rem', 
              fontWeight: '700', 
              color: 'var(--text-primary, #1e293b)',
              margin: 0
            }}>
              Something went wrong
            </h2>
            <p style={{ 
              color: 'var(--text-secondary, #64748b)',
              fontSize: '1rem',
              maxWidth: '400px',
              lineHeight: '1.5'
            }}>
              This component failed to load properly. We've been notified and are looking into it.
            </p>
          </div>

          <button
            onClick={() => window.location.reload()}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              padding: '0.75rem 1.5rem',
              backgroundColor: 'var(--accent-color, #10b981)',
              color: 'white',
              border: 'none',
              borderRadius: '12px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'transform 0.2s, box-shadow 0.2s',
              boxShadow: '0 4px 6px -1px rgba(16, 185, 129, 0.2)'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 10px -1px rgba(16, 185, 129, 0.3)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(16, 185, 129, 0.2)';
            }}
          >
            <FaRedo />
            Refresh Application
          </button>
          
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <details style={{ 
              marginTop: '1rem', 
              textAlign: 'left', 
              width: '100%',
              fontSize: '0.875rem',
              color: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.05)',
              padding: '1rem',
              borderRadius: '8px',
              border: '1px dashed #ef4444'
            }}>
              <summary style={{ cursor: 'pointer', fontWeight: '600' }}>Error Details (Dev Only)</summary>
              <pre style={{ marginTop: '0.5rem', whiteSpace: 'pre-wrap', overflowX: 'auto' }}>
                {this.state.error.toString()}
                <br />
                {this.state.errorInfo?.componentStack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
