import React from 'react';
import './Button.css';

const Button = ({
  children,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon: Icon,
  className = '',
  type = 'button',
  ...props
}) => {
  const isDisabled = disabled || loading;
  const buttonClasses = `
    btn
    btn-${variant}
    btn-${size}
    ${loading ? 'btn-loading' : ''}
    ${isDisabled ? 'btn-disabled' : ''}
    ${className}
  `.trim();

  return (
    <button
      className={buttonClasses}
      disabled={isDisabled}
      aria-disabled={isDisabled}
      aria-busy={loading}
      type={type}
      {...props}
    >
      {loading && <span className="btn-spinner" aria-hidden="true" />}
      {Icon && <Icon className="btn-icon" aria-hidden="true" />}
      <span className="btn-text">{children}</span>
    </button>
  );
};

export default Button;
