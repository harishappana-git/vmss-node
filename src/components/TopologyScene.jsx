import React, { useEffect, useMemo } from 'react';
import PropTypes from 'prop-types';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Line, PerspectiveCamera } from '@react-three/drei';

const NODE_COLORS = {
  'distributed-system': '#58a6ff',
  server: '#f78166',
  device: '#f2cc60',
  grid: '#8ddb8c',
  block: '#d2a8ff',
  warp: '#79c0ff',
  thread: '#ffa657'
};

const NODE_SCALE = {
  'distributed-system': 2.2,
  server: 2,
  device: 1.8,
  grid: 1.4,
  block: 1.2,
  warp: 0.9,
  thread: 0.65
};

const NodeMesh = ({ node, position, onSelect, isActiveChild }) => {
  const radius = NODE_SCALE[node.type] || 1;
  const color = NODE_COLORS[node.type] || '#cccccc';

  return (
    <group position={position}>
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          onSelect(node);
        }}
        scale={isActiveChild ? 1.05 : 1}
      >
        <sphereGeometry args={[radius, 48, 48]} />
        <meshStandardMaterial
          color={color}
          opacity={isActiveChild ? 0.95 : 0.8}
          transparent
          roughness={0.4}
          metalness={0.3}
        />
      </mesh>
      <Html center distanceFactor={8} position={[0, radius + 0.75, 0]}>
        <div
          style={{
            padding: '0.25rem 0.5rem',
            borderRadius: '999px',
            background: 'rgba(13, 17, 23, 0.8)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            whiteSpace: 'nowrap',
            fontSize: '0.75rem'
          }}
        >
          {node.name}
        </div>
      </Html>
    </group>
  );
};

NodeMesh.propTypes = {
  node: PropTypes.object.isRequired,
  position: PropTypes.arrayOf(PropTypes.number).isRequired,
  onSelect: PropTypes.func.isRequired,
  isActiveChild: PropTypes.bool
};

NodeMesh.defaultProps = {
  isActiveChild: false
};

const ActiveNodeContent = ({ node, onSelectChild }) => {
  const children = node.children ?? [];
  const radius = Math.max(6, children.length * 0.75 + 4);

  return (
    <group>
      <NodeMesh node={node} position={[0, 0, 0]} onSelect={onSelectChild} />
      {children.map((child, index) => {
        const angle = (index / children.length) * Math.PI * 2;
        const distance = radius + Math.sin(index) * 0.2;
        const x = Math.cos(angle) * distance;
        const z = Math.sin(angle) * distance;
        return (
          <group key={child.id}>
            <Line
              points={[[0, 0, 0], [x, 0, z]]}
              color="#3fb0ff"
              lineWidth={1}
              transparent
              opacity={0.35}
            />
            <NodeMesh
              node={child}
              position={[x, 0, z]}
              onSelect={onSelectChild}
              isActiveChild
            />
          </group>
        );
      })}
    </group>
  );
};

ActiveNodeContent.propTypes = {
  node: PropTypes.object.isRequired,
  onSelectChild: PropTypes.func.isRequired
};

const CameraRig = ({ focusDepth }) => {
  const { camera } = useThree();

  useEffect(() => {
    const targetZ = 18 + focusDepth * 4;
    const targetY = 8 + focusDepth * 1.5;
    camera.position.lerp({ x: 0, y: targetY, z: targetZ }, 0.2);
    camera.updateProjectionMatrix();
  }, [focusDepth, camera]);

  return null;
};

CameraRig.propTypes = {
  focusDepth: PropTypes.number.isRequired
};

const Scene = ({ topology, activeNode, focusDepth, onSelectNode }) => (
  <Canvas shadows dpr={[1, 2]}>
    <PerspectiveCamera makeDefault position={[0, 12, 28]} fov={48} />
    <color attach="background" args={[0x05070a]} />
    <ambientLight intensity={0.3} />
    <directionalLight position={[12, 18, 16]} intensity={1.1} castShadow />
    <pointLight position={[-8, 6, -10]} intensity={0.6} color="#58a6ff" />
    <pointLight position={[10, -6, 4]} intensity={0.3} color="#d2a8ff" />

    <group position={[0, 0, 0]}>
      <ActiveNodeContent node={activeNode} onSelectChild={onSelectNode} />
    </group>

    <OrbitControls
      enablePan
      enableDamping
      dampingFactor={0.2}
      minDistance={6}
      maxDistance={64}
      minPolarAngle={0.15 * Math.PI}
      maxPolarAngle={0.85 * Math.PI}
    />
    <CameraRig focusDepth={focusDepth} />
  </Canvas>
);

Scene.propTypes = {
  topology: PropTypes.object.isRequired,
  activeNode: PropTypes.object.isRequired,
  focusDepth: PropTypes.number.isRequired,
  onSelectNode: PropTypes.func.isRequired
};

const TopologyScene = ({ topology, activeNode, focusDepth, onSelectNode }) => {
  const memoisedScene = useMemo(
    () => (
      <Scene
        key={activeNode.id}
        topology={topology}
        activeNode={activeNode}
        focusDepth={focusDepth}
        onSelectNode={onSelectNode}
      />
    ),
    [topology, activeNode, focusDepth, onSelectNode]
  );

  return memoisedScene;
};

TopologyScene.propTypes = {
  topology: PropTypes.object.isRequired,
  activeNode: PropTypes.object.isRequired,
  focusDepth: PropTypes.number.isRequired,
  onSelectNode: PropTypes.func.isRequired
};

export default TopologyScene;
