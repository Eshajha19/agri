import React from 'react';
import { FaExclamationTriangle, FaHome, FaRedo } from 'react-icons/fa';
import Button from './Button';
import './ImprovedErrorBoundary.css';

class ImprovedErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: null,
    };
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      error,
      errorId: Math.random().toString(36).substr(2, 9),
    };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });

    if (process.env.NODE_ENV === 'production') {
      this.logErrorToService(error, errorInfo);
    } else {
      console.error('Error caught by boundary:', error, errorInfo);
    }
  }

  logErrorToService = (error, errorInfo) => {
    try {
      const errorLog = {
        message: error.toString(),
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        errorId: this.state.errorId,
      };

      fetch('/api/log-error', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorLog),
      }).catch(() => {});
    } catch (err) {
      console.error('Failed to log error:', err);
    }
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
    });
  };

  handleHome = () => {
    window.location.href = '/';
  };

  toggleDetails = () => {
    this.setState((prev) => ({ showDetails: !prev.showDetails }));
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-container">
          <div className="error-boundary-content">
            <div className="error-icon">
              <FaExclamationTriangle />
            </div>

            <h1 className="error-title">Oops! Something Went Wrong</h1>

            <p className="error-description">
              We're sorry for the inconvenience. An unexpected error occurred while processing your request.
            </p>

            {this.state.errorId && (
              <div className="error-id">
                Error ID: <code>{this.state.errorId}</code>
              </div>
            )}

            <div className="error-actions">
              <Button
                variant="primary"
                icon={FaRedo}
                onClick={this.handleReset}
              >
                Try Again
              </Button>

              <Button
                variant="secondary"
                icon={FaHome}
                onClick={this.handleHome}
              >
                Go Home
              </Button>
            </div>

            {process.env.NODE_ENV !== 'production' && (
              <>
                <button
                  className="error-details-toggle"
                  onClick={this.toggleDetails}
                >
                  {this.state.showDetails ? 'Hide' : 'Show'} Error Details
                </button>

                {this.state.showDetails && (
                  <div className="error-details">
                    {this.state.error && (
                      <div className="error-section">
                        <h3>Error Message</h3>
                        <pre>{this.state.error.toString()}</pre>
                      </div>
                    )}

                    {this.state.error?.stack && (
                      <div className="error-section">
                        <h3>Stack Trace</h3>
                        <pre>{this.state.error.stack}</pre>
                      </div>
                    )}

                    {this.state.errorInfo?.componentStack && (
                      <div className="error-section">
                        <h3>Component Stack</h3>
                        <pre>{this.state.errorInfo.componentStack}</pre>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}

            <div className="error-help">
              <p>
                If this problem persists, please contact our support team or try again later.
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ImprovedErrorBoundary;
