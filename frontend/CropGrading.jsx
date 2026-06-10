import { useState, useRef, useEffect } from "react";
import { BarChart3, Award, Sprout, Droplets, Sun, Thermometer, Scale } from "lucide-react";

export default function CropGrading({ onClose }) {
  const [cropType, setCropType] = useState("wheat");
  const [weight, setWeight] = useState("");
  const [moisture, setMoisture] = useState("");
  const [protein, setProtein] = useState("");
  const [isOrganic, setIsOrganic] = useState(false);
  const mountedRef = useRef(true);
  const gradingRequestRef = useRef(0);
  const gradingInProgressRef = useRef(false);

  const cropTypes = [
    { value: "wheat", label: "Wheat", icon: "🌾" },
    { value: "rice", label: "Rice", icon: "🍚" },
    { value: "maize", label: "Maize", icon: "🌽" },
    { value: "cotton", label: "Cotton", icon: "🧵" },
    { value: "sugarcane", label: "Sugarcane", icon: "🎋" },
  ];

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      gradingRequestRef.current++;
    };
  }, []);

  const handleGrade = () => {
    if (!weight) return;

    if (gradingInProgressRef.current) {
      return;
    }

    gradingInProgressRef.current = true;

    const requestId = ++gradingRequestRef.current;

    setLoading(true);
    
    setTimeout(() => {
      const moistureNum = parseFloat(moisture) || 0;
      const proteinNum = parseFloat(protein) || 0;
      
      let grade = "C";
      let score = 50;
      
      if (cropType === "wheat") {
        if (proteinNum >= 12 && moistureNum <= 12) {
          grade = "A";
          score = 90;
        } else if (proteinNum >= 10 && moistureNum <= 14) {
          grade = "B";
          score = 75;
        } else if (proteinNum >= 8 && moistureNum <= 16) {
          grade = "C";
          score = 60;
        } else {
          score = 40;
        }
      } else if (cropType === "rice") {
        if (moistureNum <= 14) {
          grade = "A";
          score = 90;
        } else if (moistureNum <= 16) {
          grade = "B";
          score = 75;
        } else {
          grade = "C";
          score = 60;
        }
      } else {
        if (moistureNum <= 15) {
          grade = "A";
          score = 85;
        } else if (moistureNum <= 18) {
          grade = "B";
          score = 70;
        } else {
          grade = "C";
          score = 55;
        }
      }
      
      if (isOrganic) score = Math.min(100, score + 5);
      if (
        !mountedRef.current ||
        requestId !== gradingRequestRef.current
      ) {
        gradingInProgressRef.current = false;
        return;
      }
      
      setResult({
        grade,
        score,
        moisture: moistureNum,
        protein: proteinNum,
        weight: weight,
        organicBonus: isOrganic,
      });
      gradingInProgressRef.current = false;
      setLoading(false);
    }, 1000);
  };

  const getGradeColor = (grade) => {
    if (grade === "A") return "#16a34a";
    if (grade === "B") return "#f59e0b";
    return "#ef4444";
  };

  const getGradeDescription = (grade) => {
    if (grade === "A") return "Premium grade - Excellent quality for export";
    if (grade === "B") return "Good grade - Suitable for domestic markets";
    return "Standard grade - Suitable for local markets";
  };

  return (
    <div style={{
      maxWidth: "500px",
      margin: "40px auto",
      padding: "24px",
      background: "#fff",
      borderRadius: "16px",
      boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
      position: "relative"
    }}>
      <button
        className="close-btn"
        onClick={onClose}
        aria-label="Close"
      >
        ✕
      </button>

      <h2 style={{ color: "#16a34a", fontSize: "24px", marginBottom: "20px" }}>
        📊 Crop Grading Assistant
      </h2>

      <div style={{ marginBottom: "16px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontWeight: "500" }}>
          Crop Type
        </label>
        <select
          value={cropType}
          onChange={(e) => {
            setCropType(e.target.value);
            setResult(null);
            gradingRequestRef.current++;
          }}
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #d1d5db",
            fontSize: "14px"
          }}
        >
          {cropTypes.map(crop => (
            <option key={crop.value} value={crop.value}>
              {crop.icon} {crop.label}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "16px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontWeight: "500" }}>
          Weight (kg)
        </label>
        <input
          type="number"
          value={weight}
          onChange={(e) => setWeight(e.target.value)}
          placeholder="Enter total weight harvested"
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #d1d5db",
            fontSize: "14px"
          }}
        />
      </div>

      <div style={{ marginBottom: "16px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontWeight: "500" }}>
          Moisture Content (%)
        </label>
        <input
          type="number"
          value={moisture}
          onChange={(e) => setMoisture(e.target.value)}
          placeholder="e.g., 12.5"
          step="0.1"
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #d1d5db",
            fontSize: "14px"
          }}
        />
      </div>

      {cropType === "wheat" && (
        <div style={{ marginBottom: "16px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontWeight: "500" }}>
            Protein Content (%)
          </label>
          <input
            type="number"
            value={protein}
            onChange={(e) => setProtein(e.target.value)}
            placeholder="e.g., 11.5"
            step="0.1"
            style={{
              width: "100%",
              padding: "10px",
              borderRadius: "8px",
              border: "1px solid #d1d5db",
              fontSize: "14px"
            }}
          />
        </div>
      )}

      <div style={{ marginBottom: "20px" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={isOrganic}
            onChange={(e) => setIsOrganic(e.target.checked)}
          />
          <span>Organic Produce</span>
        </label>
      </div>

      <button
        onClick={handleGrade}
        disabled={!weight || loading}
        style={{
          width: "100%",
          padding: "12px",
          backgroundColor: loading ? "#86efac" : "#16a34a",
          color: "white",
          border: "none",
          borderRadius: "8px",
          fontSize: "16px",
          cursor: !weight || loading ? "not-allowed" : "pointer",
          opacity: !weight || loading ? 0.7 : 1
        }}
      >
        {loading ? "⏳ Analyzing..." : "📈 Calculate Grade"}
      </button>

      {result && (
        <div style={{
          marginTop: "24px",
          padding: "20px",
          background: "#f0fdf4",
          borderRadius: "12px",
          border: "1px solid #bbf7d0"
        }}>
          <div style={{ textAlign: "center", marginBottom: "16px" }}>
            <div style={{
              fontSize: "48px",
              fontWeight: "bold",
              color: getGradeColor(result.grade),
              marginBottom: "8px"
            }}>
              Grade {result.grade}
            </div>
            <div style={{ fontSize: "16px", color: "#555" }}>
              {getGradeDescription(result.grade)}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            <div style={{ padding: "12px", background: "white", borderRadius: "8px" }}>
              <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>Quality Score</div>
              <div style={{ fontSize: "24px", fontWeight: "bold", color: "#111" }}>
                {result.score}/100
              </div>
            </div>
            <div style={{ padding: "12px", background: "white", borderRadius: "8px" }}>
              <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>Estimated Value</div>
              <div style={{ fontSize: "24px", fontWeight: "bold", color: "#111" }}>
                ₹{(parseFloat(result.weight) * (result.grade === "A" ? 30 : result.grade === "B" ? 20 : 15)).toLocaleString()}
              </div>
            </div>
          </div>

          {result.organicBonus && (
            <div style={{
              marginTop: "12px",
              padding: "8px",
              background: "#dcfce7",
              borderRadius: "6px",
              fontSize: "12px",
              color: "#166534",
              textAlign: "center"
            }}>
              🌱 Organic bonus applied (+5 points)
            </div>
          )}

          <div style={{ marginTop: "16px", fontSize: "14px", color: "#555" }}>
            <p><strong>Recommendations:</strong></p>
            <ul style={{ margin: "8px 0", paddingLeft: "20px" }}>
              <li>Store in cool, dry conditions</li>
              <li>Use within 6 months for optimal quality</li>
              <li>Contact local procurement centers for premium rates</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}