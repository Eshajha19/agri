import React from "react";
import "./Legal.css";


export default function Privacy() {
  return (
    <div className="legal-page">
      <h1>Privacy Policy</h1>
      <p className="last-updated">Last Updated: April 2026</p>

      <section>
        <h2>1. Introduction</h2>
        <p>
          At Fasal Saathi, we are committed to protecting your privacy. This Privacy
          Policy explains how we collect, use, and safeguard your information when
          you use our agricultural advisory platform.
        </p>
      </section>

      <section>
        <h2>2. Information We Collect</h2>
        <p>We collect the following types of information:</p>
        <ul>
          <li><strong>Farm Data:</strong> Soil conditions, crop types, location information</li>
          <li><strong>Usage Data:</strong> How you interact with our platform</li>
          <li><strong>Device Information:</strong> Browser type, IP address, device characteristics</li>
          <li><strong>Communication Data:</strong> Messages sent through our chat system</li>
        </ul>
      </section>

      <section>
        <h2>3. How We Use Your Information</h2>
        <p>Your information is used to:</p>
        <ul>
          <li>Provide personalized agricultural recommendations</li>
          <li>Improve our AI models and platform functionality</li>
          <li>Ensure platform security and prevent misuse</li>
          <li>Communicate important updates and features</li>
        </ul>
      </section>

      <section>
        <h2>4. Data Storage and Security</h2>
        <p>
          We use Firebase for secure data storage and implement industry-standard
          security measures to protect your information. Data is encrypted in transit
          and at rest.
        </p>
      </section>

      <section>
        <h2>5. Data Sharing</h2>
        <p>
          We do not sell your personal information to third parties. Data may be
          shared only for platform functionality (e.g., with AI services) or as
          required by law.
        </p>
      </section>

      <section>
        <h2>6. Your Rights</h2>
        <p>You have the right to:</p>
        <ul>
          <li>Access your personal data</li>
          <li>Request data deletion</li>
          <li>Opt-out of non-essential communications</li>
          <li>Request data portability</li>
        </ul>
      </section>

      <section>
        <h2>7. Cookies and Tracking</h2>
        <p>
          We use cookies to enhance your experience and analyze platform usage.
          You can control cookie preferences through your browser settings.
        </p>
      </section>

      <section>
        <h2>8. Children's Privacy</h2>
        <p>
          Our platform is not intended for children under 13. We do not knowingly
          collect personal information from children under 13.
        </p>
      </section>

      <section>
        <h2>9. Changes to This Policy</h2>
        <p>
          We may update this Privacy Policy from time to time. Significant changes
          will be communicated through the platform.
        </p>
      </section>

      <section>
        <h2>10. Contact Us</h2>
        <p>
          If you have questions about this Privacy Policy, please contact us via
          the Contact page or through our support channels.
        </p>
      </section>
    </div>
  );
}