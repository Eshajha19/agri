import React from 'react';
import {
  FaFolderOpen,
  FaSearch,
  FaLock,
  FaExclamationTriangle,
  FaLeaf,
} from 'react-icons/fa';
import './EmptyState.css';

const EmptyState = ({
  type = 'empty',
  title = 'No Data',
  message = 'There is no data to display',
  actionText = 'Get Started',
  onAction = null,
  icon = null,
  className = '',
}) => {
  const icons = {
    empty: <FaFolderOpen className="empty-state-icon" />,
    search: <FaSearch className="empty-state-icon" />,
    locked: <FaLock className="empty-state-icon" />,
    error: <FaExclamationTriangle className="empty-state-icon" />,
    farming: <FaLeaf className="empty-state-icon" />,
  };

  const customIcon = icon || icons[type] || icons.empty;

  return (
    <div className={`empty-state ${className}`}>
      <div className="empty-state-content">
        <div className="empty-state-icon-wrapper">{customIcon}</div>
        <h2 className="empty-state-title">{title}</h2>
        <p className="empty-state-message">{message}</p>
        {onAction && (
          <button className="empty-state-action" onClick={onAction}>
            {actionText}
          </button>
        )}
      </div>
    </div>
  );
};

export default EmptyState;
