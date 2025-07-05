
import React, { useRef, useMemo, useEffect, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { MathUtils } from "three";
import vertexShader from "../shaders/vertexShader";
import fragmentShader from "../shaders/fragmentShader";

const Blob = () => {
  const mesh = useRef();
  const hover = useRef(false);
  const [scrollPercent, setScrollPercent] = useState(0); // Scroll percentage

  const uniforms = useMemo(() => ({
    u_time: { value: 0 },
    u_intensity: { value: 0.3 },
  }), []);

  // Track scroll position and convert to a percentage (0 to 1)
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scroll = Math.min(scrollTop / docHeight, 1); // clamp between 0 and 1
      setScrollPercent(scroll);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useFrame((state) => {
    const { clock } = state;

    if (mesh.current) {
      // Animate shader time and hover
      mesh.current.material.uniforms.u_time.value = 0.4 * clock.getElapsedTime();
      mesh.current.material.uniforms.u_intensity.value = MathUtils.lerp(
        mesh.current.material.uniforms.u_intensity.value,
        hover.current ? 1 : 0.15,
        0.02
      );

      // Animate horizontal movement to the right
      const rightLimit = 3; // max x position
      const targetX = MathUtils.lerp(0, rightLimit, scrollPercent);
      mesh.current.position.x = MathUtils.lerp(mesh.current.position.x, targetX, 0.05);
    }
  });

  return (
    <mesh
      ref={mesh}
      scale={1.5}
      position={[0, 0, 0]}
      onPointerOver={() => (hover.current = true)}
      onPointerOut={() => (hover.current = false)}
    >
      <icosahedronGeometry args={[2, 20]} />
      <shaderMaterial
        attach="material"
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
      />
    </mesh>
  );
};

export default Blob;
