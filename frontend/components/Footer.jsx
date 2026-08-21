import React from "react";
import { useTranslation } from 'react-i18next';
import { Link } from "react-router-dom";
import { FaSeedling, FaPhoneAlt, FaGlobe } from "react-icons/fa";
import "./Footer.css";

const Footer = () => {
  const { t } = useTranslation();
  return (
    <footer className="global-footer" role="contentinfo">
        <div className="footer-content">
          <div className="footer-grid">
            <div className="footer-section">
              <div className="footer-brand">
                <FaSeedling className="footer-logo" aria-hidden="true" />
                <span className="notranslate" translate="no">Fasal Saathi</span>
              </div>
              <p className="footer-description">
                {t("footer.description", "AI-powered agricultural advisor helping farmers with crop planning, weather insights, irrigation, and yield optimization.")}
              </p>
              <div className="footer-contact">
                <FaPhoneAlt aria-hidden="true" />
                <span>+91 98765 43210</span>
              </div>
            </div>
            <div className="footer-section">
              <h4 id="quick-links-heading">{t("footer.quickLinks", "Quick Links")}</h4>
              <nav aria-labelledby="quick-links-heading">
                <Link to="/" aria-label={t("footer.homeAria", "Go to Home Page")}>{t("nav.home")}</Link>
                <Link to="/advisor" aria-label={t("footer.advisorAria", "Consult the AI Advisor")}>{t("nav.advisor")}</Link>
                <Link to="/how-it-works" aria-label={t("footer.howItWorksAria", "How Fasal Saathi helps you")}>{t("nav.howItWorks")}</Link>
                <Link to="/schemes" aria-label={t("footer.schemesAria", "View Government Schemes for farmers")}>{t("schemes.title")}</Link>
                <Link to="/dashboard" aria-label={t("footer.dashboardAria", "Go to your farming dashboard")}>{t("nav.dashboard", "Dashboard")}</Link>
                <Link to="/calendar" aria-label={t("footer.calendarAria", "View your farming activity calendar")}>{t("footer.activityCalendar", "Activity Calendar")}</Link>
                <Link to="/market-prices" aria-label={t("footer.marketPricesAria", "Check latest market prices for crops")}>{t("marketPrices.title")}</Link>
                <Link to="/community" aria-label={t("footer.communityAria", "Join the community discussion")}>{t("menu.community", "Community")}</Link>
                <Link to="/share-feedback" aria-label={t("footer.feedbackAria", "Share your thoughts with us")}>{t("footer.shareFeedback", "Share Feedback")}</Link>
              </nav>
            </div>
            <div className="footer-section">
              <h4 id="resources-heading">{t("nav.resources")}</h4>
              <nav aria-labelledby="resources-heading">
                <Link to="/crop-guide" aria-label={t("footer.cropGuideAria", "View the Crop Guide")}>{t("nav.cropGuide")}</Link>
                <Link to="/weather" aria-label={t("footer.weatherAria", "Check weather updates")}>{t("footer.weatherUpdates", "Weather Updates")}</Link>
                <Link to="/soil-analysis" aria-label={t("footer.soilAnalysisAria", "Get soil analysis insights")}>{t("footer.soilAnalysis", "Soil Analysis")}</Link>
                <Link to="/faq" aria-label={t("footer.faqAria", "Frequently Asked Questions")}>{t("footer.faqs", "FAQs")}</Link>
              </nav>
            </div>
            <div className="footer-section">
              <h4 id="company-heading">{t("footer.company", "Company")}</h4>
              <nav aria-labelledby="company-heading">
                <Link to="/about" aria-label={t("footer.aboutAria", "Learn about Fasal Saathi")}>{t("nav.about")}</Link>
                <Link to="/contact" aria-label={t("footer.contactAria", "Contact our support team")}>{t("menu.contact", "Contact")}</Link>
                <Link to="/privacy-policy" aria-label={t("footer.privacyAria", "Read our Privacy Policy")}>{t("footer.privacyPolicy", "Privacy Policy")}</Link>
                <Link to="/terms" aria-label={t("footer.termsAria", "Read our Terms of Service")}>{t("footer.termsOfService", "Terms of Service")}</Link>
              </nav>
            </div>
          </div>
          <div className="footer-bottom">
            <div className="footer-socials">
              <FaGlobe aria-hidden="true" />
              <span>{t("footer.availableAcrossIndia", "Available Across India")}</span>
            </div>
            <p className="footer-copyright">
              © 2026 <span className="notranslate" translate="no">Fasal Saathi</span>. {t("footer.rights", "All rights reserved. MIT Licensed.")}
            </p>
          </div>
        </div>
      </footer>
    );
};

export default Footer;
