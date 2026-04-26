import React from "react";
import "./Legal.css";


export default function Terms() {
  return (
    <div className="legal-page">
      <h1>Terms of Service</h1>
      <p className="last-updated">Last Updated: April 2026</p>

      <section>
        <h2>1. Introduction</h2>
        <p>
          Welcome to Fasal Saathi. By using our platform, you agree to comply
          with and be bound by these Terms of Service. Please read them carefully.
        </p>
      </section>

      <section>
        <h2>2. Use of the Platform</h2>
        <p>
          Fasal Saathi provides AI-based agricultural recommendations. The
          information provided is for guidance only and should not be considered
          a substitute for professional agricultural advice.
        </p>
      </section>

      <section>
        <h2>3. User Responsibilities</h2>
        <ul>
          <li>Provide accurate farm and soil data</li>
          <li>Use the platform ethically</li>
          <li>Do not misuse or attempt to disrupt services</li>
        </ul>
      </section>

      <section>
        <h2>4. Data & Privacy</h2>
        <p>
          We collect and process data as described in our Privacy Policy. Your
          data is handled securely using trusted services like Firebase.
        </p>
      </section>

      <section>
        <h2>5. Limitation of Liability</h2>
        <p>
          Fasal Saathi is not liable for any losses or damages resulting from the
          use of recommendations provided by the platform.
        </p>
      </section>

      <section>
        <h2>6. Changes to Terms</h2>
        <p>
          We may update these Terms from time to time. Continued use of the
          platform implies acceptance of updated terms.
        </p>
      </section>

      <section>
        <h2>7. Contact Us</h2>
        <p>
          If you have any questions about these Terms, please contact us via the
          Contact page.
        </p>
      </section>
    </div>
  );
}