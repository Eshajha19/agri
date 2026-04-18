import React, { useState, useRef, useEffect } from "react";

function LanguageDropdown({ options, value, onChange }) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const containerRef = useRef(null);

  const selected = options.find((o) => o.value === value);

  // Search works by both English name (value) and native label
  const filtered = options.filter(
    (o) =>
      o.label.toLowerCase().includes(search.toLowerCase()) ||
      o.value.toLowerCase().includes(search.toLowerCase())
  );

  // Selected language always appears at top
  const sorted = [
    ...filtered.filter((o) => o.value === value),
    ...filtered.filter((o) => o.value !== value),
  ];

  // Close on outside click
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
        setSearch("");
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, []);

  const handleSelect = (val) => {
    onChange(val);
    setIsOpen(false);
    setSearch("");
  };

  return (
    <div className="lang-dropdown notranslate" ref={containerRef}>
      <button
        type="button"
        className="lang-dropdown-btn"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        {selected ? selected.label : "Language"} ▾
      </button>

      {isOpen && (
        <div className="lang-dropdown-panel" role="listbox">
          <input
            type="text"
            className="lang-dropdown-search"
            placeholder="Search language..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            autoFocus
          />

          <ul className="lang-dropdown-list">
            {sorted.length > 0 ? (
              sorted.map((o) => (
                <li
                  key={o.value}
                  role="option"
                  aria-selected={o.value === value}
                  className={`lang-dropdown-item ${
                    o.value === value ? "lang-dropdown-item--selected" : ""
                  }`}
                  onClick={() => handleSelect(o.value)}
                >
                  {o.label}
                  {o.value === value && (
                    <span className="lang-dropdown-check">✓</span>
                  )}
                </li>
              ))
            ) : (
              <li className="lang-dropdown-empty">No results found</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

export default LanguageDropdown;