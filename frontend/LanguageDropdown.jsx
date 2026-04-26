import React, { useState, useRef, useEffect } from "react";

function LanguageDropdown({ options, value, onChange }) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const containerRef = useRef(null);
  const listRef = useRef(null);
  const searchRef = useRef(null);

  const selected = options.find((o) => o.value === value);

  const filtered = options.filter(
  (o) =>
    o.label.toLowerCase().includes(search.toLowerCase()) ||
    o.value.toLowerCase().includes(search.toLowerCase()) ||
    (o.englishName && o.englishName.toLowerCase().includes(search.toLowerCase()))
);

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
        setFocusedIndex(-1);
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, []);

  // Reset focused index when search changes
  useEffect(() => {
    setFocusedIndex(-1);
  }, [search]);

  const handleSelect = (val) => {
    onChange(val);
    setIsOpen(false);
    setSearch("");
    setFocusedIndex(-1);
  };

  const handleButtonKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setIsOpen((prev) => !prev);
    } else if (e.key === "Escape") {
      setIsOpen(false);
      setSearch("");
    }
  };

  const handleSearchKeyDown = (e) => {
    if (e.key === "Escape") {
      setIsOpen(false);
      setSearch("");
      setFocusedIndex(-1);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocusedIndex((i) => Math.min(i + 1, sorted.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && focusedIndex >= 0 && sorted.length > 0) {
      e.preventDefault();
      handleSelect(sorted[focusedIndex].value);
    }
  };

  const handleItemKeyDown = (e, val) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleSelect(val);
    }
  };

  return (
    <div className="lang-dropdown notranslate" ref={containerRef}>
      <button
        type="button"
        className="lang-dropdown-btn"
        onClick={() => setIsOpen((prev) => !prev)}
        onKeyDown={handleButtonKeyDown}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls="lang-dropdown-list"
        aria-label="Select language"
      >
        {selected ? selected.label : "Language"} ▾
      </button>

      {isOpen && (
        <div className="lang-dropdown-panel">
          <input
            ref={searchRef}
            type="text"
            className="lang-dropdown-search"
            placeholder="Search language..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            aria-label="Search language"
            autoFocus
          />

          <ul
            id="lang-dropdown-list"
            ref={listRef}
            className="lang-dropdown-list"
            role="listbox"
            aria-label="Languages"
          >
            {sorted.length > 0 ? (
              sorted.map((o, index) => (
                <li
                  key={o.value}
                  role="option"
                  aria-selected={o.value === value}
                  tabIndex={0}
                  className={`lang-dropdown-item ${
                    o.value === value ? "lang-dropdown-item--selected" : ""
                  } ${focusedIndex === index ? "lang-dropdown-item--focused" : ""}`}
                  onClick={() => handleSelect(o.value)}
                  onKeyDown={(e) => handleItemKeyDown(e, o.value)}
                >
                  {o.label}
                  {o.value === value && (
                    <span className="lang-dropdown-check" aria-hidden="true">✓</span>
                  )}
                </li>
              ))
            ) : (
              <li className="lang-dropdown-empty" role="option" aria-selected="false">
                No results found
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

export default LanguageDropdown;