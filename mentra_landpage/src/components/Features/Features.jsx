import React, { useEffect } from 'react';
import "./Features.css"; 


const featuresRow1 = [
  { icon: "🤖", title: "Empathetic Chatbot", desc: "Engage with an emotionally aware chatbot designed to understand, validate, and support your emotional state." },
  { icon: "📊", title: "Emotion Analysis", desc: "Detects sadness, fear, anger, and more — adapting tone and responses to support your needs." },
  { icon: "🔁", title: "Dynamic Flow", desc: "Conversation strategy evolves with you — from greeting to crisis support." },
  { icon: "🧠", title: "Configurable Behavior", desc: "Users or admins can tune empathy levels, tone (supportive, casual, pro), and conversation depth." }
];

const featuresRow2 = [
  { icon: "🎙️", title: "Multimodal Input", desc: "Supports text input now — with future expansion for voice and image support." },
  { icon: "🖼️", title: "Visual Diagnostics", desc: "Upload visible symptoms (e.g., skin rash). Get AI-powered feedback with up to 79% accuracy." },
  { icon: "⚠️", title: "Crisis Detection", desc: "Detects extreme distress and triggers crisis handling protocols with care." },
  { icon: "📁", title: "Document Ingestion", desc: "Ingests and learns from mental health docs, expanding its ability to assist." }
];

const featuresRow3 = [
  { icon: "🔐", title: "Privacy First", desc: "Your conversations are encrypted and secure. We prioritize your mental health privacy above all." },
  { icon: "📱", title: "Cross-Platform", desc: "Access mentra from web, mobile, or desktop. Your support system follows you everywhere." },
  { icon: "🎯", title: "Personalized Care", desc: "AI learns your patterns and preferences to provide increasingly personalized mental health support." },
  { icon: "🌍", title: "24/7 Availability", desc: "Mental health doesn't follow business hours. Get support whenever you need it, day or night." }
];

const Feature = () => {
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes ripple {
        to {
          transform: scale(2);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const track = entry.target.querySelector('.carousel-track');
        if (track) {
          track.style.animationPlayState = entry.isIntersecting ? 'running' : 'paused';
        }
      });
    }, {
      root: null,
      rootMargin: '100px',
      threshold: 0.1,
    });

    document.querySelectorAll('.carousel-row').forEach(row => observer.observe(row));

    return () => {
      document.querySelectorAll('.carousel-row').forEach(row => observer.unobserve(row));
    };
  }, []);

  const handleRipple = (e) => {
    const card = e.currentTarget;
    const ripple = document.createElement('span');
    const rect = card.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;

    ripple.style.cssText = `
      position: absolute;
      border-radius: 50%;
      background: rgba(0, 212, 255, 0.3);
      width: ${size}px;
      height: ${size}px;
      left: ${x}px;
      top: ${y}px;
      animation: ripple 0.8s ease-out;
      pointer-events: none;
      z-index: 1000;
    `;

    card.appendChild(ripple);
    setTimeout(() => ripple.remove(), 800);
  };

  const renderRow = (data, direction = 'left', slower = false) => (
    <div className="carousel-row">
      <div
        className="carousel-track"
        style={{
          animation: `${direction === 'left' ? 'scrollLeft' : 'scrollRight'} ${slower ? '40s' : '20s'} linear infinite`
        }}
      >
        {[...data, ...data].map((item, index) => (
          <div className="card" key={index} onClick={handleRipple}>
            <h3>{item.icon} {item.title}</h3>
            <p>{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="Features">
      <h1>Key Features of mentra</h1>
      {renderRow(featuresRow1, 'left')}
      {renderRow(featuresRow2, 'right')}
      {renderRow(featuresRow3, 'left', true)}
    </div>
  );
};

export default Feature;



