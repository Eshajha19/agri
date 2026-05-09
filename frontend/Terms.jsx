import React from "react";
import "./Legal.css";

export default function Terms() {
  return (
    <div className="legal-page">
      <h1>
        <span className="notranslate">Terms of Service</span>
      </h1>

      <p className="last-updated">Last Updated: April 2026</p>

      <section className="legal-section">
        <h2>1. Introduction</h2>

        <p>
          Welcome to <span className="notranslate">Fasal Saathi</span>. By using
          our platform, you agree to comply with these Terms of Service.
        </p>
      </section>

      <section className="legal-section">
        <h2>2. Use of the Platform</h2>

        <p>
          <span className="notranslate">Fasal Saathi</span> provides AI-based
          agricultural recommendations.
        </p>
      </section>

      <section className="legal-section">
        <h2>3. User Responsibilities</h2>

        <ul>
          <li>Provide accurate farm and soil data</li>
          <li>Use the platform ethically</li>
          <li>Do not misuse the services</li>
        </ul>
      </section>

      <section className="legal-section">
        <h2>4. Data & Privacy</h2>

        <p>
          We collect and process data as described in our Privacy Policy.
        </p>
      </section>

      <section className="legal-section">
        <h2>5. Limitation of Liability</h2>

        <p>
          <span className="notranslate">Fasal Saathi</span> is not liable for
          losses resulting from platform recommendations.
        </p>
      </section>

      <section className="legal-section">
        <h2>6. Changes to Terms</h2>

        <p>
          We may update these Terms from time to time.
        </p>
      </section>

      <section className="legal-section">
        <h2>
          7. <span className="notranslate">Contact Us</span>
        </h2>

        <p>
          Contact us through the Contact page for any questions.
        </p>
      </section>
    </div>
  );
}