import React from 'react';
import './LoadingSkeleton.css';

export const SkeletonLoader = ({ width = '100%', height = '20px', borderRadius = '4px', className = '' }) => (
  <div className={`skeleton-loader ${className}`} style={{ width, height, borderRadius }} />
);

export const CardSkeleton = ({ count = 3 }) => (
  <div className="skeleton-card-grid">
    {Array.from({ length: count }).map((_, i) => (
      <div key={i} className="skeleton-card">
        <SkeletonLoader height="200px" borderRadius="8px" />
        <div className="skeleton-card-content">
          <SkeletonLoader height="20px" width="70%" />
          <SkeletonLoader height="16px" width="100%" className="mt-2" />
          <SkeletonLoader height="16px" width="90%" className="mt-1" />
        </div>
      </div>
    ))}
  </div>
);

export const TableSkeleton = ({ rows = 5, columns = 4 }) => (
  <div className="skeleton-table">
    <div className="skeleton-table-header">
      {Array.from({ length: columns }).map((_, i) => (
        <SkeletonLoader key={i} height="20px" width="100%" />
      ))}
    </div>
    {Array.from({ length: rows }).map((_, i) => (
      <div key={i} className="skeleton-table-row">
        {Array.from({ length: columns }).map((_, j) => (
          <SkeletonLoader key={j} height="18px" width="100%" />
        ))}
      </div>
    ))}
  </div>
);

export const ListSkeleton = ({ count = 5 }) => (
  <div className="skeleton-list">
    {Array.from({ length: count }).map((_, i) => (
      <div key={i} className="skeleton-list-item">
        <SkeletonLoader height="40px" width="40px" borderRadius="50%" />
        <div className="skeleton-list-content">
          <SkeletonLoader height="20px" width="60%" />
          <SkeletonLoader height="16px" width="80%" className="mt-2" />
        </div>
      </div>
    ))}
  </div>
);

export const DashboardSkeleton = () => (
  <div className="skeleton-dashboard">
    <div className="skeleton-dashboard-header">
      <SkeletonLoader height="40px" width="200px" />
      <SkeletonLoader height="40px" width="100px" />
    </div>
    <div className="skeleton-dashboard-grid">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="skeleton-stat-card">
          <SkeletonLoader height="16px" width="60%" />
          <SkeletonLoader height="32px" width="80%" className="mt-3" />
        </div>
      ))}
    </div>
    <div className="skeleton-dashboard-chart">
      <SkeletonLoader height="300px" />
    </div>
  </div>
);

const LoadingSkeleton = ({ type = 'spinner', count = 3, rows = 5, columns = 4 }) => {
  switch (type) {
    case 'card':
      return <CardSkeleton count={count} />;
    case 'table':
      return <TableSkeleton rows={rows} columns={columns} />;
    case 'list':
      return <ListSkeleton count={count} />;
    case 'dashboard':
      return <DashboardSkeleton />;
    default:
      return <SkeletonLoader />;
  }
};

export default LoadingSkeleton;
