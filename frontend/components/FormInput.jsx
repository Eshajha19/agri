import React, { useState } from 'react';
import { FaCheck, FaTimes, FaExclamationCircle } from 'react-icons/fa';
import './FormInput.css';

const FormInput = ({
  label,
  type = 'text',
  placeholder = '',
  value = '',
  onChange = () => {},
  error = null,
  success = false,
  disabled = false,
  required = false,
  helpText = null,
  icon: Icon = null,
  className = '',
  maxLength = null,
  pattern = null,
  autoComplete = 'off',
  ...props
}) => {
  const [focused, setFocused] = useState(false);

  const hasError = error && error.length > 0;
  const showSuccess = success && !hasError;

  const inputClasses = `
    form-input
    ${focused ? 'form-input-focused' : ''}
    ${hasError ? 'form-input-error' : ''}
    ${showSuccess ? 'form-input-success' : ''}
    ${disabled ? 'form-input-disabled' : ''}
    ${Icon ? 'form-input-with-icon' : ''}
    ${className}
  `.trim();

  return (
    <div className="form-group">
      {label && (
        <label className="form-label">
          {label}
          {required && <span className="form-required">*</span>}
        </label>
      )}

      <div className="form-input-wrapper">
        {Icon && <Icon className="form-input-icon" />}
        <input
          className={inputClasses}
          type={type}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          disabled={disabled}
          required={required}
          maxLength={maxLength}
          pattern={pattern}
          autoComplete={autoComplete}
          {...props}
        />
        {hasError && <FaTimes className="form-input-status form-input-status-error" />}
        {showSuccess && <FaCheck className="form-input-status form-input-status-success" />}
      </div>

      {hasError && (
        <div className="form-feedback form-feedback-error">
          <FaExclamationCircle className="form-feedback-icon" />
          {error}
        </div>
      )}

      {helpText && !hasError && (
        <div className="form-feedback form-feedback-help">
          {helpText}
        </div>
      )}

      {maxLength && (
        <div className="form-char-count">
          {value.length} / {maxLength}
        </div>
      )}
    </div>
  );
};

export default FormInput;
