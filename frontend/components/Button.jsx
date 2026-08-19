import React from 'react';
import { useTranslation } from 'react-i18next';
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
  const { t } = useTranslation();
  const buttonClasses = `
    btn
    btn-${variant}
    btn-${size}
    ${loading ? 'btn-loading' : ''}
    ${disabled ? 'btn-disabled' : ''}
    ${className}
  `.trim();

  return (
    <button
      className={buttonClasses}
      disabled={disabled || loading}
      type={type}
      {...props}
    >
      {loading && <span className="btn-spinner" />}
      {Icon && <Icon className="btn-icon" />}
      <span className="btn-text">{children}</span>
    </button>
  );
};

export default Button;
