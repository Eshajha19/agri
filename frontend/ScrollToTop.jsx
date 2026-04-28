import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { FaArrowUp } from "react-icons/fa";

function ScrollToTop({ threshold = 280 }) {
  const { pathname, hash } = useLocation();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [pathname]);

  useEffect(() => {
    const handleScroll = () => {
      setVisible(window.scrollY > threshold);
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [threshold]);

  useEffect(() => {
    if (!hash) return;
    const targetId = hash.replace("#", "");
    const targetElement = document.getElementById(targetId) || document.querySelector(hash);
    if (targetElement) {
      targetElement.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [hash]);

  const handleClick = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <button
      className={`scroll-to-top ${visible ? "show" : ""}`}
      onClick={handleClick}
      aria-label="Back to top"
      type="button"
    >
      <FaArrowUp />
    </button>
  );
}

export default ScrollToTop;
