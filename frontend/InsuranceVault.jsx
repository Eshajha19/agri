import React, { useState } from "react";

function InsuranceVault({ documents }) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");

  const filteredDocs = documents.filter(doc =>
    (category ? doc.category === category : true) &&
    (search ? doc.name.toLowerCase().includes(search.toLowerCase()) : true)
  );

  return (
    <div className="vault-container">
      <h2>Insurance Document Vault</h2>

      <div className="controls">
        <input
          type="text"
          placeholder="Search documents..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select onChange={(e) => setCategory(e.target.value)}>
          <option value="">All Categories</option>
          <option value="Policy Certificates">Policy Certificates</option>
          <option value="Claim Receipts">Claim Receipts</option>
          <option value="Survey Reports">Survey Reports</option>
          <option value="Compensation Records">Compensation Records</option>
          <option value="Land Ownership Documents">Land Ownership Documents</option>
        </select>
      </div>

      <div className="doc-list">
        {filteredDocs.map(doc => (
          <div key={doc.id} className="doc-card">
            <p><strong>{doc.name}</strong></p>
            <p>Category: {doc.category}</p>
            <p>Uploaded: {doc.uploadedAt}</p>
            <button onClick={() => window.open(doc.url)}>Preview</button>
            <button onClick={() => downloadFile(doc.url)}>Download</button>
          </div>
        ))}
      </div>
    </div>
  );
}

function downloadFile(url) {
  const link = document.createElement("a");
  link.href = url;
  link.download = url.split("/").pop();
  link.click();
}

export default InsuranceVault;
