import React, { useState, useMemo, useCallback, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Grid,
  Plane,
  Box,
  Sphere,
  Cylinder,
  PerspectiveCamera,
  ContactShadows,
  Environment
} from "@react-three/drei";

import * as THREE from "three";
import {
  Save,
  Trash2,
  Box as BoxIcon,
  TreePine,
  Droplets,
  Home,
  Grid as GridIcon,
  Info,
  Download
} from "lucide-react";

import "./FarmPlanner3D.css";

/* ---------------- CONFIG ---------------- */

const OBJECT_TYPES = {
  CROP: { color: "#4caf50" },
  TREE: { color: "#2e7d32" },
  WATER: { color: "#2196f3" },
  BUILDING: { color: "#795548" }
};

/* ---------------- 3D OBJECT ---------------- */

const FarmObject = React.memo(({ item, selected, onSelect }) => {
  const isSelected = selected === item.id;

  const model = useMemo(() => {
    switch (item.type) {
      case "TREE":
        return (
          <group>
            <Cylinder args={[0.1, 0.1, 0.6]} position={[0, 0.3, 0]}>
              <meshStandardMaterial color="#5d4037" />
            </Cylinder>
            <Sphere args={[0.4]} position={[0, 0.8, 0]}>
              <meshStandardMaterial color="#2e7d32" />
            </Sphere>
          </group>
        );

      case "BUILDING":
        return (
          <Box args={[1, 0.8, 1]}>
            <meshStandardMaterial color="#8d6e63" />
          </Box>
        );

      case "WATER":
        return (
          <Cylinder args={[0.5, 0.5, 0.8]}>
            <meshStandardMaterial color="#2196f3" metalness={0.5} />
          </Cylinder>
        );

      default:
        return (
          <Box args={[1, 0.1, 1]}>
            <meshStandardMaterial color="#4caf50" />
          </Box>
        );
    }
  }, [item.type]);

  return (
    <group position={item.position} onClick={(e) => { e.stopPropagation(); onSelect(item.id); }}>
      {model}

      {isSelected && (
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.6, 0.75, 32]} />
          <meshBasicMaterial color="yellow" />
        </mesh>
      )}
    </group>
  );
});

/* ---------------- MAIN APP ---------------- */

export default function FarmPlanner3D() {
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState(null);
  const [activeTool, setActiveTool] = useState("CROP");

  /* ---------------- SAFE ADD (NO ROUNDING BUG) ---------------- */

  const addItem = useCallback((e) => {
    if (!e?.point) return;

    const pos = new THREE.Vector3().copy(e.point);
    pos.y = 0.1;

    const newItem = {
      id: Date.now(),
      type: activeTool,
      position: [pos.x, pos.y, pos.z]
    };

    setItems(prev => [...prev, newItem]);
    setSelected(newItem.id);
  }, [activeTool]);

  /* ---------------- DELETE ---------------- */

  const deleteSelected = () => {
    setItems(prev => prev.filter(i => i.id !== selected));
    setSelected(null);
  };

  /* ---------------- EXPORT SAFE ---------------- */

  const exportLayout = useCallback(() => {
    const blob = new Blob([JSON.stringify(items, null, 2)], {
      type: "application/json"
    });

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "farm-layout.json";
    a.click();

    URL.revokeObjectURL(url); // 🔥 memory cleanup
  }, [items]);

  /* ---------------- RENDER ---------------- */

  return (
    <div className="farm-planner-container">

      {/* SIDEBAR */}
      <div className="planner-sidebar">
        <h2><GridIcon /> Farm 3D Planner</h2>

        {Object.keys(OBJECT_TYPES).map(type => (
          <button
            key={type}
            onClick={() => setActiveTool(type)}
            className={activeTool === type ? "active" : ""}
          >
            {type}
          </button>
        ))}

        <button onClick={deleteSelected} disabled={!selected}>
          <Trash2 /> Delete
        </button>

        <button onClick={exportLayout}>
          <Download /> Export
        </button>
      </div>

      {/* 3D VIEW */}
      <div className="planner-viewport">
        <Canvas shadows>

          <PerspectiveCamera makeDefault position={[10, 10, 10]} />
          <OrbitControls />

          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 10, 5]} intensity={1} />

          <Grid infiniteGrid />

          {/* CLICK PLANE */}
          <Plane
            args={[100, 100]}
            rotation={[-Math.PI / 2, 0, 0]}
            onClick={addItem}
          >
            <meshStandardMaterial transparent opacity={0} />
          </Plane>

          <Suspense fallback={null}>
            {items.map(item => (
              <FarmObject
                key={item.id}
                item={item}
                selected={selected}
                onSelect={setSelected}
              />
            ))}
          </Suspense>

          <ContactShadows opacity={0.4} scale={20} />

        </Canvas>
      </div>
    </div>
  );
}