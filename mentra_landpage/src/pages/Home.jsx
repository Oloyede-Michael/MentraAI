import React from "react";
import { Canvas } from "@react-three/fiber";
import Blob from '../components/Blob';
import "../App.css";
import NavBar from "../components/NavBar";
import HeroSection from "../components/HeroSection";
import HomeBody from "../components/HomeBody";
import Body from "../components/Body";
import Features from "../components/Features/Features";
import FAQ from "../components/FAQ/FAQ";
import Howitworks from "../components/HowItWorks/HowITWorks";

const Home = () => {
  return (
    <div className="wholeContent">
      <div style={{ background:' linear-gradient(to right, #c3dafe, #e9d8fd)', }}>
        <HeroSection />
        <div className="BlobContainer">
          <div className="dot">
            <div className="canvas-container">
              <Canvas camera={{ position: [0.0, 0.0, 8.0] }}>
                <ambientLight intensity={0.5} />
                <directionalLight position={[2, 2, 5]} intensity={1} />
                <Blob />
              </Canvas>
            </div>
        </div>
      
          <div className="red">
          <HomeBody />
          </div>
        </div>
        <div>
        <Body />
        </div>
      </div>
      <div>
        <Howitworks />
      </div>
      <div>
         <Features />
      </div>
      <div>
        <FAQ />
      </div>
      
    </div>
  );
};

export default Home;

