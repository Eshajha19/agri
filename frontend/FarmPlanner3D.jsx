import React, { useState, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Plane, Box, Sphere, Cylinder, PerspectiveCamera, ContactShadows, Environment, Text } from '@react-three/drei';
import * as THREE from 'three';
import { Save, Trash2, Box as BoxIcon, TreePine, Droplets, Home, Grid as GridIcon, Info, Download } from 'lucide-react';
import './FarmPlanner3D.css';

const OBJECT_TYPES = {
  CROP: { name: 'Crop Plot', color: '#4caf50', geometry: 'box', scale: [1, 0.2, 1] },
  TREE: { name: 'Orchard Tree', color: '#2e7d32', geometry: 'cylinder', scale: [0.3, 1, 0.3] },
  WATER: { name: 'Irrigation Tank', color: '#2196f3', geometry: 'cylinder', scale: [0.8, 0.8, 0.8] },
  BUILDING: { name: 'Farm House', color: '#795548', geometry: 'box', scale: [1.2, 1, 1.2] }
};

function DraggableObject({ position, type, isSelected, onSelect }) {
  const [hovered, setHover] = useState(false);
  const color = isSelected ? '#ffeb3b' : (hovered ? '#fff176' : '#ffffff');

  // Realistic 3D Models
  const renderModel = () => {
    switch (type) {
      case 'TREE':
        return (
          <group>
            <Cylinder args={[0.1, 0.1, 0.6]} position={[0, 0.3, 0]}>
              <meshStandardMaterial color="#5d4037" />
            </Cylinder>
            <Sphere args={[0.4, 16, 16]} position={[0, 0.8, 0]}>
              <meshStandardMaterial color="#2e7d32" />
            </Sphere>
          </group>
        );
      case 'BUILDING':
        return (
          <group>
            <Box args={[1.2, 0.8, 1.2]} position={[0, 0.4, 0]}>
              <meshStandardMaterial color="#8d6e63" />
            </Box>
            <Cylinder args={[0, 1, 0.6, 4]} position={[0, 1.1, 0]} rotation={[0, Math.PI / 4, 0]}>
              <meshStandardMaterial color="#b71c1c" />
            </Cylinder>
          </group>
        );
      case 'WATER':
        return (
          <group>
            <Cylinder args={[0.5, 0.5, 0.8, 32]} position={[0, 0.4, 0]}>
              <meshStandardMaterial color="#455a64" metalness={0.6} roughness={0.2} />
            </Cylinder>
            <Cylinder args={[0.55, 0.55, 0.1, 32]} position={[0, 0.85, 0]}>
              <meshStandardMaterial color="#263238" />
            </Cylinder>
          </group>
        );
      case 'CROP':
      default:
        return (
          <group>
            <Box args={[1, 0.1, 1]} position={[0, 0.05, 0]}>
              <meshStandardMaterial color="#3e2723" />
            </Box>
            <Box args={[0.9, 0.15, 0.9]} position={[0, 0.1, 0]}>
              <meshStandardMaterial color="#4caf50" />
            </Box>
          </group>
        );
    }
  };

  return (
    <group 
      position={position} 
      onPointerOver={() => setHover(true)}
      onPointerOut={() => setHover(false)}
      onClick={(e) => { e.stopPropagation(); onSelect(); }}
    >
      {renderModel()}
      {isSelected && (
        <mesh position={[0, -0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.7, 0.8, 32]} />
          <meshBasicMaterial color="#ffeb3b" />
        </mesh>
      )}
    </group>
  );
}

export default function FarmPlanner3D() {
  const [items, setItems] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [activeTool, setActiveTool] = useState('CROP');
  const [gridSize, setGridSize] = useState(10);

  const addItem = (e) => {
    // Basic Raycasting projection for placement (simplified)
    const newId = Date.now();
    const newItem = {
      id: newId,
      type: activeTool,
      position: [Math.round(e.point.x), OBJECT_TYPES[activeTool].scale[1] / 2, Math.round(e.point.z)],
      color: OBJECT_TYPES[activeTool].color
    };
    setItems([...items, newItem]);
    setSelectedId(newId);
  };

  const deleteSelected = () => {
    setItems(items.filter(item => item.id !== selectedId));
    setSelectedId(null);
  };

  const exportLayout = () => {
    const data = JSON.stringify(items);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'my-farm-layout.json';
    a.click();
  };

  return (
    <div className="farm-planner-container">
      <div className="planner-sidebar">
        <div className="sidebar-header">
          <h2><GridIcon size={20} /> 3D Planner</h2>
          <p>Optimize your land usage</p>
        </div>

        <div className="tool-section">
          <label>Add Elements</label>
          <div className="tool-grid">
            {Object.keys(OBJECT_TYPES).map(type => (
              <button 
                key={type}
                className={`tool-btn ${activeTool === type ? 'active' : ''}`}
                onClick={() => setActiveTool(type)}
                title={OBJECT_TYPES[type].name}
              >
                {type === 'CROP' && <BoxIcon size={20} />}
                {type === 'TREE' && <TreePine size={20} />}
                {type === 'WATER' && <Droplets size={20} />}
                {type === 'BUILDING' && <Home size={20} />}
                <span>{OBJECT_TYPES[type].name}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="action-section">
          <button className="action-btn delete" onClick={deleteSelected} disabled={!selectedId}>
            <Trash2 size={18} /> Delete Selected
          </button>
          <button className="action-btn save" onClick={exportLayout}>
            <Download size={18} /> Export JSON
          </button>
        </div>

        <div className="planner-info">
          <Info size={16} />
          <p>Click on the grid to place objects. Use mouse to rotate/zoom.</p>
        </div>
      </div>

      <div className="planner-viewport">
        <Canvas shadows>
          <PerspectiveCamera makeDefault position={[10, 10, 10]} fov={50} />
          <OrbitControls makeDefault minPolarAngle={0} maxPolarAngle={Math.PI / 2.1} />
          
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} castShadow />
          <Environment preset="city" />

          {/* Ground Grid */}
          <Grid 
            infiniteGrid 
            fadeDistance={50} 
            fadeStrength={5} 
            cellSize={1} 
            sectionSize={5} 
            sectionColor="#2e7d32"
            cellColor="#999"
          />
          
          {/* Clickable Plane for object placement */}
          <Plane 
            args={[100, 100]} 
            rotation={[-Math.PI / 2, 0, 0]} 
            position={[0, -0.01, 0]} 
            onClick={addItem}
          >
            <meshStandardMaterial color="#f0fdf4" transparent opacity={0.5} />
          </Plane>

          {/* Render Items */}
          {items.map(item => (
            <DraggableObject 
              key={item.id}
              {...item}
              isSelected={selectedId === item.id}
              onSelect={() => setSelectedId(item.id)}
            />
          ))}

          <ContactShadows position={[0, 0, 0]} opacity={0.4} scale={20} blur={2} far={4.5} />
        </Canvas>
        
        <div className="viewport-overlay">
          <div className="stats-badge">
            Total Elements: {items.length}
          </div>
        </div>
      </div>
    </div>
  );
}