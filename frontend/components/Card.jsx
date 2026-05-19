import React from 'react';
import './Card.css';

const Card = ({
  children,
  variant = 'default',
  elevation = 'medium',
  hoverable = false,
  onClick = null,
  className = '',
  header = null,
  footer = null,
  image = null,
  imageAlt = '',
  ...props
}) => {
  const cardClasses = `
    card
    card-${variant}
    card-elevation-${elevation}
    ${hoverable ? 'card-hoverable' : ''}
    ${onClick ? 'card-clickable' : ''}
    ${className}
  `.trim();

  return (
    <div
      className={cardClasses}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyPress={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick();
        }
      }}
      {...props}
    >
      {image && (
        <div className="card-image">
          <img src={image} alt={imageAlt} />
        </div>
      )}

      {header && <div className="card-header">{header}</div>}

      <div className="card-content">{children}</div>

      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
};

export default Card;
