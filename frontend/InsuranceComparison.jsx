import React, { useState } from "react";

function InsuranceComparison({ policies }) {
  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState("premiumCost");

  const filteredPolicies = policies
    .filter(p => filter ? p.supportedCrops.includes(filter) : true)
    .sort((a, b) => a[sortKey] > b[sortKey] ? 1 : -1);

  return (
    <div className="comparison-container">
      <h2>Insurance Policy Comparison Center</h2>

      <div className="controls">
        <select onChange={e => setFilter(e.target.value)}>
          <option value="">All Crops</option>
          <option value="Wheat">Wheat</option>
          <option value="Rice">Rice</option>
        </select>

        <select onChange={e => setSortKey(e.target.value)}>
          <option value="premiumCost">Premium Cost</option>
          <option value="coverageAmount">Coverage Amount</option>
          <option value="claimSettlementTime">Claim Settlement Time</option>
        </select>
      </div>

      <table className="comparison-table">
        <thead>
          <tr>
            <th>Policy</th>
            <th>Premium</th>
            <th>Coverage</th>
            <th>Claim Time</th>
            <th>Supported Crops</th>
            <th>Subsidy</th>
            <th>Recommended</th>
          </tr>
        </thead>
        <tbody>
          {filteredPolicies.map(p => (
            <tr key={p.id} className={p.recommended ? "highlight" : ""}>
              <td>{p.name}</td>
              <td>₹{p.premiumCost}</td>
              <td>₹{p.coverageAmount}</td>
              <td>{p.claimSettlementTime}</td>
              <td>{p.supportedCrops.join(", ")}</td>
              <td>{p.subsidyAvailable ? "Yes" : "No"}</td>
              <td>{p.recommended ? "⭐" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default InsuranceComparison;
