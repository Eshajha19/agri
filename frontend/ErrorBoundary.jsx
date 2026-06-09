import React, { useState } from 'react';
import { FaExclamationTriangle, FaRedo, FaTimesCircle, FaCheckCircle } from 'react-icons/fa';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
      retryCount: 0,
      isRecovering: false,
      showFeedbackForm: false,
      userFeedback: '',
      reproductionSteps: '',
      recoveryAttempts: 0,
      maxRetries: props.maxRetries || 3,
      retryDelay: props.retryDelay || 2000,
    };
    this.feedbackFormRef = React.createRef();
  }

  static getDerivedStateFromError(error) {
    const errorId = `error-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return {
      hasError: true,
      error,
      errorId,
      retryCount: 0
    };
  }

  componentDidCatch(error, errorInfo) {
    const { errorId } = this.state;
    this.setState({ errorInfo });

    this.recordErrorAnalytics(error, errorInfo);
    this.logStructuredError(error, errorInfo, errorId);

    if (this.props.onError) {
      this.props.onError({ error, errorInfo, errorId });
    }
  }

  recordErrorAnalytics(error, errorInfo) {
    try {
      const analytics = window.__errorAnalytics || {};
      const errorType = error.name || 'Unknown';

      if (!analytics.errorCounts) analytics.errorCounts = {};
      if (!analytics.affectedComponents) analytics.affectedComponents = {};

      analytics.errorCounts[errorType] = (analytics.errorCounts[errorType] || 0) + 1;
      analytics.lastErrorTime = new Date().toISOString();

      const componentMatch = errorInfo?.componentStack?.match(/in (\w+)/);
      if (componentMatch) {
        const componentName = componentMatch[1];
        analytics.affectedComponents[componentName] = (analytics.affectedComponents[componentName] || 0) + 1;
      }

      window.__errorAnalytics = analytics;

      if (this.props.onAnalytics) {
        this.props.onAnalytics(analytics);
      }
    } catch (e) {
      console.warn('Failed to record error analytics:', e);
    }
  }

  logStructuredError(error, errorInfo, errorId) {
    const structuredLog = {
      error_id: errorId,
      timestamp: new Date().toISOString(),
      message: error.toString(),
      type: error.name || 'UnknownError',
      stack_trace: error.stack,
      component_stack: errorInfo?.componentStack,
      browser_info: {
        userAgent: navigator.userAgent,
        language: navigator.language,
        viewport: `${window.innerWidth}x${window.innerHeight}`
      },
      environment: process.env.NODE_ENV
    };

    console.error('[Structured Error Log]', JSON.stringify(structuredLog, null, 2));

    if (this.props.onStructuredLog) {
      this.props.onStructuredLog(structuredLog);
    }
  }

  handleRetry = async () => {
    const { retryCount, maxRetries, retryDelay, recoveryAttempts } = this.state;

    if (retryCount >= maxRetries) {
      this.setState({ isRecovering: false });
      return;
    }

    this.setState({ isRecovering: true, recoveryAttempts: recoveryAttempts + 1 });

    const exponentialDelay = retryDelay * Math.pow(2, retryCount);
    await new Promise(resolve => setTimeout(resolve, exponentialDelay));

    this.setState(prevState => ({
      hasError: false,
      error: null,
      errorInfo: null,
      isRecovering: false,
      retryCount: prevState.retryCount + 1
    }));
  };

  toggleFeedbackForm = () => {
    this.setState(prevState => ({
      showFeedbackForm: !prevState.showFeedbackForm
    }));
  };

  handleSubmitFeedback = () => {
    const { userFeedback, reproductionSteps, errorId, error } = this.state;

    if (!userFeedback.trim()) {
      alert('Please provide feedback before submitting');
      return;
    }

    const feedbackData = {
      error_id: errorId,
      timestamp: new Date().toISOString(),
      user_feedback: userFeedback,
      reproduction_steps: reproductionSteps || 'Not provided',
      error_message: error?.toString(),
      user_agent: navigator.userAgent
    };

    console.log('[User Feedback]', JSON.stringify(feedbackData, null, 2));

    if (this.props.onFeedback) {
      this.props.onFeedback(feedbackData);
    }

    this.setState({
      showFeedbackForm: false,
      userFeedback: '',
      reproductionSteps: '',
      feedbackSubmitted: true
    });

    setTimeout(() => this.setState({ feedbackSubmitted: false }), 3000);
  };

  getErrorCategory(error) {
    const message = error?.toString() || '';
    if (message.includes('Network') || message.includes('fetch')) return 'Network error';
    if (message.includes('Firestore') || message.includes('database')) return 'Database error';
    if (message.includes('401') || message.includes('Unauthorized')) return 'Permission error';
    if (message.includes('TypeError')) return 'Rendering error';
    return 'Unexpected error';
  }

  getSuggestedActions(error) {
    const category = this.getErrorCategory(error);
    const actions = {
      'Network error': [
        'Check your internet connection',
        'Wait a moment and try again',
        'If the problem persists, contact support'
      ],
      'Database error': [
        'Try refreshing the page',
        'Clear your browser cache',
        'Contact support if the issue continues'
      ],
      'Permission error': [
        'Check that you have the necessary permissions',
        'Log out and log back in',
        'Contact an administrator if you believe this is an error'
      ],
      'Rendering error': [
        'Refresh the application',
        'Clear your browser cache and cookies',
        'Use a different browser to verify the issue'
      ]
    };
    return actions[category] || actions['Unexpected error'];
  }

  render() {
    const {
      hasError,
      error,
      errorInfo,
      errorId,
      retryCount,
      maxRetries,
      isRecovering,
      showFeedbackForm,
      userFeedback,
      reproductionSteps,
      feedbackSubmitted
    } = this.state;

    if (hasError) {
      const errorCategory = this.getErrorCategory(error);
      const suggestedActions = this.getSuggestedActions(error);
      const isRecoverable = retryCount < maxRetries;

      return (
        <div style={{
          padding: '2rem 1.5rem',
          textAlign: 'center',
          background: 'var(--bg-secondary, #f8fafc)',
          borderRadius: '16px',
          border: '1px solid var(--border-color, #e2e8f0)',
          margin: '2rem auto',
          maxWidth: '700px',
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

          <div style={{ gap: '0.5rem', display: 'flex', flexDirection: 'column', width: '100%' }}>
            <h2 style={{
              fontSize: '1.5rem',
              fontWeight: '700',
              color: 'var(--text-primary, #1e293b)',
              margin: 0
            }}>
              {errorCategory}
            </h2>
            <p style={{
              color: 'var(--text-secondary, #64748b)',
              fontSize: '1rem',
              maxWidth: '100%',
              lineHeight: '1.5',
              margin: 0
            }}>
              {error?.message || 'An unexpected error occurred'}
            </p>
            <p style={{
              color: '#64748b',
              fontSize: '0.875rem',
              margin: '0.5rem 0 0 0',
              fontFamily: 'monospace',
              wordBreak: 'break-all'
            }}>
              Error ID: {errorId}
            </p>
          </div>

          <div style={{
            textAlign: 'left',
            width: '100%',
            padding: '1rem',
            backgroundColor: 'rgba(59, 130, 246, 0.05)',
            borderRadius: '8px',
            border: '1px solid rgba(59, 130, 246, 0.2)'
          }}>
            <p style={{ margin: '0 0 0.5rem 0', fontWeight: '600', color: '#1e293b' }}>
              Suggested actions:
            </p>
            <ol style={{ margin: 0, paddingLeft: '1.5rem', color: '#64748b', fontSize: '0.95rem' }}>
              {suggestedActions.map((action, idx) => (
                <li key={idx} style={{ marginBottom: '0.25rem' }}>{action}</li>
              ))}
            </ol>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', width: '100%', flexWrap: 'wrap', justifyContent: 'center' }}>
            {isRecoverable && (
              <button
                onClick={this.handleRetry}
                disabled={isRecovering}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  backgroundColor: isRecovering ? '#cbd5e1' : 'var(--accent-color, #10b981)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  fontWeight: '600',
                  cursor: isRecovering ? 'not-allowed' : 'pointer',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  opacity: isRecovering ? 0.7 : 1
                }}
              >
                <FaRedo />
                {isRecovering ? `Retrying... (${retryCount}/${maxRetries})` : 'Try Again'}
              </button>
            )}

            <button
              onClick={() => window.location.reload()}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.5rem',
                backgroundColor: '#64748b',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'transform 0.2s, box-shadow 0.2s'
              }}
            >
              <FaRedo />
              Reload Page
            </button>

            <button
              onClick={this.toggleFeedbackForm}
              style={{
                padding: '0.75rem 1.5rem',
                backgroundColor: 'transparent',
                color: 'var(--accent-color, #10b981)',
                border: '2px solid var(--accent-color, #10b981)',
                borderRadius: '12px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Send Feedback
            </button>
          </div>

          {feedbackSubmitted && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              color: '#10b981',
              borderRadius: '8px',
              fontSize: '0.9rem',
              animation: 'fadeInOut 3s ease-in-out'
            }}>
              <FaCheckCircle /> Thank you for your feedback!
            </div>
          )}

          {showFeedbackForm && (
            <div style={{
              width: '100%',
              padding: '1rem',
              backgroundColor: '#f1f5f9',
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
              ref: this.feedbackFormRef
            }}>
              <textarea
                value={userFeedback}
                onChange={(e) => this.setState({ userFeedback: e.target.value })}
                placeholder="Describe what you were trying to do when this error occurred..."
                style={{
                  padding: '0.75rem',
                  border: '1px solid #cbd5e1',
                  borderRadius: '6px',
                  fontFamily: 'inherit',
                  fontSize: '0.95rem',
                  minHeight: '80px',
                  resize: 'vertical'
                }}
              />
              <textarea
                value={reproductionSteps}
                onChange={(e) => this.setState({ reproductionSteps: e.target.value })}
                placeholder="Steps to reproduce (optional)..."
                style={{
                  padding: '0.75rem',
                  border: '1px solid #cbd5e1',
                  borderRadius: '6px',
                  fontFamily: 'inherit',
                  fontSize: '0.95rem',
                  minHeight: '60px',
                  resize: 'vertical'
                }}
              />
              <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                <button
                  onClick={() => this.setState({ showFeedbackForm: false })}
                  style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: '#e2e8f0',
                    color: '#1e293b',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: '500'
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={this.handleSubmitFeedback}
                  style={{
                    padding: '0.5rem 1.5rem',
                    backgroundColor: 'var(--accent-color, #10b981)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: '600'
                  }}
                >
                  Submit Feedback
                </button>
              </div>
            </div>
          )}

          {process.env.NODE_ENV === 'development' && error && (
            <details style={{
              marginTop: '0.5rem',
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
                {error.toString()}
                <br />
                {errorInfo?.componentStack}
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
