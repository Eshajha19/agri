import React, { useMemo, useState } from "react";
import { getPestInfo } from "./utils/pestDatabase";
import { pestSeasonalData } from "./utils/pestSeasonalData";
import { useTranslation } from "react-i18next";

const formatPestLabel = (pestKey) =>
  pestKey
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const monthOrder = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

export default function PestCalendar() {
  const [selectedRegion, setSelectedRegion] = useState("All");
  const [selectedCrop, setSelectedCrop] = useState("All");
  const [selectedMonth, setSelectedMonth] = useState("All");
  const { i18n } = useTranslation();

  const regionOptions = useMemo(
    () => ["All", ...new Set(pestSeasonalData.map((entry) => entry.region))],
    [],
  );

  const cropOptions = useMemo(
    () => ["All", ...new Set(pestSeasonalData.map((entry) => entry.crop))],
    [],
  );

  const monthOptions = useMemo(() => {
    const usedMonths = new Set(
      pestSeasonalData.flatMap((entry) => entry.activeMonths)
    );

    return [
      "All",
      ...monthOrder.filter((month) => usedMonths.has(month))
    ];
  }, []);

  const filteredEntries = pestSeasonalData.filter((entry) => {
    const matchesRegion = selectedRegion === "All" || entry.region === selectedRegion;
    const matchesCrop = selectedCrop === "All" || entry.crop === selectedCrop;
    const matchesMonth =
      selectedMonth === "All" || entry.activeMonths.includes(selectedMonth);

    return matchesRegion && matchesCrop && matchesMonth;
  });

  const groupedEntries = useMemo(() => {
    const monthsToRender =
      selectedMonth === "All"
        ? monthOrder
        : [selectedMonth];

    return monthsToRender
      .map((month) => ({
        month,
        pests: filteredEntries.filter((entry) =>
          entry.activeMonths.includes(month)
        ),
      }))
      .filter((monthGroup) => monthGroup.pests.length > 0);
  }, [filteredEntries, selectedMonth]);

  return (
    <div style={{ padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>Pest Seasonal Attack Calendar</h1>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 12,
          marginTop: 20,
        }}
      >
        <label style={filterLabelStyle}>
          Region
          <select
            value={selectedRegion}
            onChange={(event) => setSelectedRegion(event.target.value)}
            style={selectStyle}
          >
            {regionOptions.map((option) => (
              <option
                key={option}
                value={option}
                style={optionStyle}
              >
                {option}
              </option>
            ))}
          </select>
        </label>

        <label style={filterLabelStyle}>
          Crop
          <select
            value={selectedCrop}
            onChange={(event) => setSelectedCrop(event.target.value)}
            style={selectStyle}
          >
            {cropOptions.map((option) => (
              <option
                key={option}
                value={option}
                style={optionStyle}
              >
                {option}
              </option>
            ))}
          </select>
        </label>

        <label style={filterLabelStyle}>
          Month
          <select
            value={selectedMonth}
            onChange={(event) => setSelectedMonth(event.target.value)}
            style={selectStyle}
          >
            {monthOptions.map((option) => (
              <option
                key={option}
                value={option}
                style={optionStyle}
              >
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      {groupedEntries.length > 0 ? (
        <div style={calendarGridStyle}>
          {groupedEntries.map((monthGroup) => (
            <section key={monthGroup.month} style={monthCardStyle}>
              <div style={monthHeaderStyle}>{monthGroup.month}</div>
              <div style={monthBodyStyle}>
                {monthGroup.pests.map((entry) => (
                  <article key={`${monthGroup.month}-${entry.pestKey}-${entry.crop}-${entry.region}`} style={pestItemStyle}>
                    <div style={pestTitleStyle}>{formatPestLabel(entry.pestKey)}</div>
                    <div style={pestSubtitleStyle}>Seasonal pest alert</div>
                    {(() => {
                      const pestInfo = getPestInfo(entry.pestKey, i18n.language);

                      return (
                        <>
                          <div style={detailRowStyle}>
                            <span style={detailLabelStyle}>Crop:</span>
                            <span style={detailValueStyle}>{entry.crop}</span>
                          </div>
                          <div style={detailRowStyle}>
                            <span style={detailLabelStyle}>Region:</span>
                            <span style={detailValueStyle}>{entry.region}</span>
                          </div>
                          <div style={detailRowStyle}>
                            <span style={detailLabelStyle}>Severity:</span>
                            <span style={detailValueStyle}>{entry.severity}</span>
                          </div>
                          <div style={detailRowStyle}>
                            <span style={detailLabelStyle}>Active Months:</span>
                            <span style={detailValueStyle}>{entry.activeMonths.join(", ")}</span>
                          </div>
                          <div style={detailRowStyle}>
                            <span style={detailLabelStyle}>Prevention:</span>
                            <span style={detailValueStyle}>
                              {pestInfo.prevention || "Follow integrated pest management practices"}
                            </span>
                          </div>
                        </>
                      );
                    })()}
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div style={emptyStateStyle}>No pest entries match the selected filters.</div>
      )}
    </div>
  );
}

const filterLabelStyle = {
  display: "grid",
  gap: 8,
  fontSize: 14,
  fontWeight: 600,
  color: "var(--text-primary)",
};

const selectStyle = {
  padding: "10px 12px",
  borderRadius: 8,
  border: "1px solid var(--border-color)",
  backgroundColor: "var(--bg-card)",
  color: "var(--text-primary)",
  fontSize: 14,
};

const calendarGridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
  gap: 16,
  marginTop: 24,
};

const monthCardStyle = {
  border: "1px solid #dbe4ea",
  borderRadius: 16,
  background: "linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)",
  overflow: "hidden",
  boxShadow: "0 8px 24px rgba(15, 23, 42, 0.06)",
};

const monthHeaderStyle = {
  padding: "12px 14px",
  background: "#0f766e",
  color: "#fff",
  fontWeight: 700,
  letterSpacing: "0.02em",
};

const monthBodyStyle = {
  display: "grid",
  gap: 12,
  padding: 14,
};

const pestItemStyle = {
  padding: 14,
  borderRadius: 12,
  border: "1px solid #e5e7eb",
  backgroundColor: "#fff",
};

const pestTitleStyle = {
  fontWeight: 700,
  color: "#111827",
  marginBottom: 2,
  fontSize: 17,
};

const pestSubtitleStyle = {
  fontSize: 13,
  color: "#64748b",
  marginBottom: 8,
};

const detailRowStyle = {
  display: "grid",
  gap: 2,
  marginTop: 6,
};

const detailLabelStyle = {
  fontSize: 12,
  fontWeight: 700,
  color: "#0f766e",
  textTransform: "uppercase",
  letterSpacing: "0.03em",
};

const detailValueStyle = {
  fontSize: 13,
  color: "#0f172a",
  lineHeight: 1.4,
};

const emptyStateStyle = {
  marginTop: 24,
  padding: 18,
  borderRadius: 12,
  backgroundColor: "#f8fafc",
  border: "1px dashed #cbd5e1",
  color: "#334155",
};

const optionStyle = {
  color: "#000",
};
