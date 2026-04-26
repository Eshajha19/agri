import React from "react";
import "./Policy.css";

export default function PrivacyPolicy() {
  return (
    <div className="policy-page">
      <div className="policy-hero">
        <h1>🔐 Privacy Policy</h1>
        <p>Your data, your control — we respect your privacy.</p>
      </div>

      <div className="policy-content">
        <section>
          <h2>1. Introduction</h2>
          <p>
            Fasal Saathi is committed to protecting your personal information.
            This policy explains what we collect, how we use it, and your rights.
          </p>
        </section>

        <section>
          <h2>2. Information We Collect</h2>
          <ul>
            <li>👤 Name and basic profile details</li>
            <li>🌾 Farming inputs (soil data, crop preferences)</li>
            <li>📍 Approximate location for weather insights</li>
            <li>📱 Device and usage data (for analytics & improvements)</li>
          </ul>
        </section>

        <section>
          <h2>3. How We Use Your Data</h2>
          <ul>
            <li>✔ Provide crop recommendations and alerts</li>
            <li>✔ Improve models and product experience</li>
            <li>✔ Personalize dashboard and suggestions</li>
          </ul>
        </section>

        <section>
          <h2>4. Data Security</h2>
          <p>
            We use secure technologies (e.g., Firebase Authentication and
            encrypted storage) to protect your data.
          </p>
        </section>

        <section>
          <h2>5. Data Sharing</h2>
          <p>
            We do not sell your personal data. Limited data may be shared with
            trusted services (e.g., weather APIs) strictly to enable features.
          </p>
        </section>

        <section>
          <h2>6. Your Rights</h2>
          <ul>
            <li>✔ Access and update your data</li>
            <li>✔ Request deletion of your account/data</li>
          </ul>
        </section>

        <section>
          <h2>7. Updates to This Policy</h2>
          <p>
            We may update this policy periodically. Significant changes will be
            communicated in-app.
          </p>
        </section>

        <section>
          <h2>8. Contact</h2>
          <p>
            📧 support@fasalsaathi.com <br />
            📞 +91 98765 43210
          </p>
        </section>
      </div>
    </div>
  );
}