import React from "react";
import { Info, Sprout, Wheat, Droplets, Mountain, Sun, Layers, Leaf } from "lucide-react";
import "./SoilGuide.css";

const SOIL_TYPES = [
  {
    id: "alluvial",
    name: "Alluvial Soil",
    description: "The most fertile and widespread soil in India, formed by river deposits.",
    characteristics: ["Rich in potash", "Low in phosphorus", "High fertility", "Varies from sandy loam to clay"],
    suitableCrops: ["Rice", "Wheat", "Sugarcane", "Cotton", "Jute", "Pulses"],
    color: "#d4b483",
    icon: <Layers size={32} />,
    visualIndicator: "high-fertility",
  },
  {
    id: "black",
    name: "Black Soil (Regur)",
    description: "Also known as Black Cotton Soil, it is famous for its moisture-retentive capacity.",
    characteristics: ["Self-ploughing", "Rich in iron, lime, calcium", "Highly argillaceous", "Retains moisture"],
    suitableCrops: ["Cotton", "Sugarcane", "Jowar", "Tobacco", "Wheat", "Linseed"],
    color: "#333333",
    icon: <Sun size={32} />,
    visualIndicator: "high-moisture",
  },
  {
    id: "red",
    name: "Red & Yellow Soil",
    description: "Developed on crystalline igneous rocks in areas of low rainfall.",
    characteristics: ["Red color due to iron diffusion", "Porous and friable", "Low in nitrogen & humus", "Responds well to fertilizers"],
    suitableCrops: ["Groundnut", "Millets", "Tobacco", "Pulses", "Rice", "Potatoes"],
    color: "#e63946",
    icon: <Mountain size={32} />,
    visualIndicator: "medium-fertility",
  },
  {
    id: "laterite",
    name: "Laterite Soil",
    description: "Formed due to intense leaching in areas of high rainfall and temperature.",
    characteristics: ["Acidic in nature", "Low base-exchange capacity", "Low in organic matter", "Needs heavy manuring"],
    suitableCrops: ["Cashew nuts", "Tea", "Coffee", "Rubber", "Coconut", "Arecanut"],
    color: "#bc6c25",
    icon: <Droplets size={32} />,
    visualIndicator: "low-fertility",
  },
  {
    id: "arid",
    name: "Arid / Desert Soil",
    description: "Found in western Rajasthan and parts of Gujarat with low rainfall.",
    characteristics: ["High salt content", "Low moisture", "Lack of humus", "Sandy texture"],
    suitableCrops: ["Barley", "Maize", "Millets", "Pulses", "Guar"],
    color: "#e9c46a",
    icon: <Sun size={32} />,
    visualIndicator: "low-moisture",
  },
  {
    id: "mountain",
    name: "Mountain / Forest Soil",
    description: "Occurs in the Himalayan regions where sufficient rain forest is available.",
    characteristics: ["Rich in humus", "Acidic in nature", "Varies with altitude", "Loamy and silty"],
    suitableCrops: ["Tea", "Coffee", "Spices", "Fruits", "Wheat", "Maize"],
    color: "#2a9d8f",
    icon: <Leaf size={32} />,
    visualIndicator: "high-organic",
  },
];

export default function SoilGuide() {
  return (
    <div className="soil-guide-page">
      <div className="soil-guide-header">
        <div className="header-icon">
          <Layers size={48} className="icon-pulse" />
        </div>
        <div className="header-text">
          <h1>Soil Type Education Hub</h1>
          <p>Discover India's diverse soil landscapes and optimize your crop selection for maximum agricultural yield.</p>
        </div>
      </div>

      <div className="soil-guide-grid">
        {SOIL_TYPES.map((soil) => (
          <div key={soil.id} className="soil-card" style={{ "--soil-color": soil.color }}>
            <div className="soil-card-header">
              <div className="soil-icon-wrap" style={{ backgroundColor: soil.color }}>
                {soil.icon}
              </div>
              <div className="soil-title-wrap">
                <h2>{soil.name}</h2>
                <span className={`visual-indicator ${soil.visualIndicator}`}>
                  {soil.visualIndicator.replace("-", " ")}
                </span>
              </div>
            </div>

            <div className="soil-card-body">
              <p className="soil-desc">{soil.description}</p>
              
              <div className="characteristics-section">
                <h3><Info size={16} /> Key Characteristics</h3>
                <ul>
                  {soil.characteristics.map((trait, index) => (
                    <li key={index}>{trait}</li>
                  ))}
                </ul>
              </div>

              <div className="crops-section">
                <h3><Sprout size={16} /> Suitable Crops</h3>
                <div className="crops-list">
                  {soil.suitableCrops.map((crop, index) => (
                    <span key={index} className="crop-tag">
                      <Wheat size={12} /> {crop}
                    </span>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="soil-card-footer">
              <div className="fertility-meter">
                <div className="meter-label">Fertility Level</div>
                <div className="meter-bar">
                  <div 
                    className="meter-fill" 
                    style={{ 
                      width: soil.visualIndicator.includes("high") ? "90%" : 
                             soil.visualIndicator.includes("medium") ? "60%" : "30%",
                      backgroundColor: soil.color 
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="soil-tips-section">
        <div className="tips-container">
          <h2><Leaf className="inline-icon" /> General Soil Management Tips</h2>
          <div className="tips-grid">
            <div className="tip-card">
              <Droplets className="tip-icon" />
              <h3>Proper Irrigation</h3>
              <p>Match your watering frequency to the soil's water retention capacity.</p>
            </div>
            <div className="tip-card">
              <Sprout className="tip-icon" />
              <h3>Crop Rotation</h3>
              <p>Rotate nitrogen-fixing crops (pulses) with nutrient-heavy crops (wheat/rice).</p>
            </div>
            <div className="tip-card">
              <Leaf className="tip-icon" />
              <h3>Organic Mulching</h3>
              <p>Add organic matter to improve soil structure and prevent moisture loss.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
