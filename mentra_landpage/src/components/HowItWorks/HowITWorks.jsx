import React, { useEffect, useState } from 'react';
import "./HowITWorks.css";
import phone1 from '../../assets/phone.png';
import phone2 from '../../assets/phone.png';
import phone3 from '../../assets/phone.png';

const Howitworks = () => {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const cards = document.querySelectorAll('.card');
    cards.forEach((item, index) => {
      item.style.opacity = '0';
      item.style.transform = 'translateY(30px)';
      setTimeout(() => {
        item.style.transition = 'all 0.6s ease-out';
        item.style.opacity = '1';
        item.style.transform = 'translateY(0)';
      }, index * 200);
    });
  }, []);

  const steps = [
    {
      title: 'Choose How You Feel',
      desc: 'Start by selecting your current emotional state calm, anxious, overwhelmed. Mentra guides you from there.',
      img: phone1,
    },
    {
      title: 'Receive Thoughtful Guidance',
      desc: 'Mentra adapts its tone and suggestions whether journaling prompts or calming affirmations.',
      img: phone2,
    },
    {
      title: 'Reflect and Track',
      desc: 'Mentra tracks your moods and sessions to help you understand your emotional growth over time.',
      img: phone3,
    },
  ];

  return (
    <div className='Howitworks'>
      <div className='How'>
        <h1>How it works</h1>
        <h2>
          A few guided steps can shift your day. Mentra helps you <br />
          understand, reset, and move forward with emotional clarity.
        </h2>
      </div>

      <div className='flexComponent'>
        <div className='cards'>
          {steps.map((step, index) => (
            <div
              className={`card ${index === activeIndex ? 'active' : ''}`}
              key={index}
              onClick={() => setActiveIndex(index)}
            >
              <h1>{step.title}</h1>
              <h2>{step.desc}</h2>
            </div>
          ))}
        </div>

        <div className='phone'>
          {steps.map((step, index) => (
            <img
              key={index}
              src={step.img}
              alt={`phone screen ${index + 1}`}
              className={`phone-img ${index === activeIndex ? 'active' : ''}`}
              style={{
                zIndex: index === activeIndex ? 10 : index,
                transform:
                  index === activeIndex
                    ? 'translateX(-50%) scale(1) rotate(0deg)'
                    : `translateX(-50%) scale(0.95) rotate(${(index - activeIndex) * 8}deg)`,
                filter: index === activeIndex ? 'brightness(1)' : 'brightness(0.6) blur(1px)',
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Howitworks