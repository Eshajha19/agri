import React, { useState, useEffect, useCallback } from "react";
import { AlertTriangle, X, RefreshCw, Home } from "lucide-react";
import "./AsyncErrorBoundary.css";

/**
 * AsyncErrorBoundary - Catches and recovers from async errors
 * 
 * Features:
 * - Catches unhandled promise rejections
 * - Displays user-friendly error messages
 * - Retry mechanisms with exponential backoff
 * - Error reporting to backend
 * - Auto-recovery for network errors
 */

class AsyncErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      errorId: null,
      errorMessage: "",
      errorCategory: "unknown",
      errorSeverity: "medium",
      retryCount: 0,
      maxRetries: 3,
      isRetrying: false,
      errorStack: "",
      timestamp: null,
      showDetails: false,
      showRetryWarning: false,
    };

    this.retryTimeout = null;
  }

  componentDidMount() {
    // Listen for unhandled promise rejections
    window.addEventListener("unhandledrejection", this.handleUnhandledRejection);
  }

  componentWillUnmount() {
    window.removeEventListener(
      "unhandledrejection",
      this.handleUnhandledRejection
    );

    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
    }
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      errorMessage: error.message,
      errorStack: error.stack,
      timestamp: new Date().toISOString(),
    };
  }

  componentDidCatch(error, errorInfo) {
    // Generate error ID
    const errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Classify error
    const { category, severity } = this.classifyError(error);

    this.setState({
      errorId,
      errorCategory: category,
      errorSeverity: severity,
    });

    // Report error to backend
    this.reportErrorToBackend(errorId, error, category, severity);

    // Log to console in development
    console.error("AsyncErrorBoundary caught error:", error, errorInfo);
  }

  handleUnhandledRejection = (event) => {
    const error = event.reason || new Error("Unknown error");

    this.setState({
      hasError: true,
      errorMessage: error.message || String(error),
      errorStack: error.stack || "No stack trace available",
      timestamp: new Date().toISOString(),
    });

    // Report error
    const errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const { category, severity } = this.classifyError(error);

    this.setState({
      errorId,
      errorCategory: category,
      errorSeverity: severity,
    });

    this.reportErrorToBackend(errorId, error, category, severity);
  };

  classifyError = (error) => {
    const errorStr = error.message?.toLowerCase() || "";
    const errorType = error.constructor?.name || "Unknown";

    // Network errors
    if (
      errorStr.includes("network") ||
      errorStr.includes("timeout") ||
      errorStr.includes("fetch") ||
      errorType.includes("Network")
    ) {
      return { category: "network", severity: "medium" };
    }

    // Authentication errors
    if (
      errorStr.includes("auth") ||
      errorStr.includes("unauthorized") ||
      errorStr.includes("401")
    ) {
      return { category: "authentication", severity: "high" };
    }

    // Validation errors
    if (
      errorStr.includes("validation") ||
      errorStr.includes("invalid") ||
      errorStr.includes("400")
    ) {
      return { category: "validation", severity: "low" };
    }

    // Database errors
    if (
      errorStr.includes("database") ||
      errorStr.includes("firestore") ||
      errorStr.includes("query")
    ) {
      return { category: "database", severity: "high" };
    }

    return { category: "unknown", severity: "medium" };
  };

  reportErrorToBackend = async (errorId, error, category, severity) => {
    try {
      // Require an authenticated Firebase user — the backend enforces a valid
      // ID token on /api/log-error to prevent unauthenticated log flooding.
      // Dynamically import firebase/auth to avoid a hard dependency in the
      // error boundary (which must render even before Firebase initialises).
      const { getAuth } = await import('firebase/auth');
      const user = getAuth().currentUser;
      if (!user) return;

      let idToken;
      try {
        idToken = await user.getIdToken();
      } catch {
        return; // Token refresh failed — skip silently
      }

      await fetch("/api/log-error", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${idToken}`,
        },
        body: JSON.stringify({
          message: error.message,
          source: "frontend_error_boundary",
          stack: error.stack,
          level: severity === "low" ? "warn" : "error",
        }),
      });
    } catch (err) {
      console.error("Failed to report error:", err);
    }
  };

  handleRetry = async () => {
    const { retryCount, maxRetries, errorCategory } = this.state;

    if (retryCount >= maxRetries) {
      this.setState({ showRetryWarning: true });
      return;
    }

    this.setState({ isRetrying: true });

    // Exponential backoff
    const backoffDelay = Math.pow(2, retryCount) * 1000;

    this.retryTimeout = setTimeout(() => {
      this.setState({
        hasError: false,
        retryCount: retryCount + 1,
        isRetrying: false,
      });
    }, backoffDelay);
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      errorId: null,
      errorMessage: "",
      errorCategory: "unknown",
      errorSeverity: "medium",
      retryCount: 0,
      isRetrying: false,
      errorStack: "",
      timestamp: null,
      showDetails: false,
      showRetryWarning: false,
    });
  };

  handleGoHome = () => {
    window.location.href = "/";
  };

  toggleDetails = () => {
    this.setState((prev) => ({ showDetails: !prev.showDetails }));
  };

  render() {
    const {
      hasError,
      errorId,
      errorMessage,
      errorCategory,
      errorSeverity,
      retryCount,
      maxRetries,
      isRetrying,
      errorStack,
      timestamp,
      showDetails,
    } = this.state;

    if (!hasError) {
      return this.props.children;
    }

    const severityColors = {
      low: "#4CAF50",
      medium: "#FFC107",
      high: "#FF9800",
      critical: "#F44336",
    };

    const categoryNames = {
      network: "Network Error",
      database: "Database Error",
      validation: "Validation Error",
      authentication: "Authentication Error",
      authorization: "Authorization Error",
      business_logic: "Business Logic Error",
      external_service: "External Service Error",
      unknown: "Unknown Error",
    };

    return (
      <div className="async-error-boundary" role="alert" aria-live="assertive">
        <div className="error-container">
          <div
            className="error-header"
            style={{ borderLeftColor: severityColors[errorSeverity] }}
          >
            <div className="error-icon-section">
              <AlertTriangle
                size={32}
                color={severityColors[errorSeverity]}
              />
              <div>
                <h2 className="error-title">Something Went Wrong</h2>
                <p className="error-category">
                  {categoryNames[errorCategory]}
                </p>
              </div>
            </div>
            <button
              className="error-close-btn"
              onClick={this.handleReset}
              title="Dismiss"
            >
              <X size={20} />
            </button>
          </div>

          <div className="error-content">
            <p className="error-message">{errorMessage}</p>

            {errorId && (
              <div className="error-details-line">
                <span className="label">Error ID:</span>
                <code className="error-id">{errorId}</code>
              </div>
            )}

            {timestamp && (
              <div className="error-details-line">
                <span className="label">Time:</span>
                <span>{new Date(timestamp).toLocaleString()}</span>
              </div>
            )}

            {retryCount > 0 && (
              <div className="error-details-line">
                <span className="label">Attempts:</span>
                <span>
                  {retryCount} / {maxRetries}
                </span>
              </div>
            )}

            {showDetails && errorStack && (
              <div className="error-stack-trace">
                <p className="stack-label">Stack Trace:</p>
                <pre className="stack-content">{errorStack}</pre>
              </div>
            )}

            <button
              className="details-toggle"
              onClick={this.toggleDetails}
            >
              {showDetails ? "Hide" : "Show"} Details
            </button>
          </div>

          <div className="error-actions">
            <button
              className="btn btn-retry"
              onClick={this.handleRetry}
              disabled={isRetrying || retryCount >= maxRetries}
            >
              {isRetrying ? (
                <>
                  <RefreshCw size={16} className="spinner" />
                  Retrying...
                </>
              ) : (
                <>
                  <RefreshCw size={16} />
                  Retry
                </>
              )}
            </button>

            <button
              className="btn btn-home"
              onClick={this.handleGoHome}
            >
              <Home size={16} />
              Go to Home
            </button>
          </div>

          {showRetryWarning && (
            <div className="error-warning" role="status">
              Maximum retries reached. Please contact support or refresh the
              page.
            </div>
          )}
        </div>
      </div>
    );
  }
}

export default AsyncErrorBoundary;
// Enhanced error boundary
