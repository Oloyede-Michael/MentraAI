import React from 'react';
import { Link } from 'react-router-dom';

const HeroSection = () => {
  return (
    <section className="w-full h-full text-center" style={{ paddingTop: '5em' }}>
      <h1 className="text-[3rem] black-future md:text-[4.5rem] text-black font-black leading-tight">
        TALK . LISTEN . HEAL
      </h1>
      <h2 className="text-[25px] text-center text-black titi mx-auto" style={{ padding: '1em 0.3em' }}>
        Whether you are feeling anxious or overwhelmed, Mentra is here to listen.
      </h2>

      <Link to="/login">
        <button className="cursor-pointer bg-indigo-600 text-white text-[18px] titi shadow-md hover:scale-125 delay-75 duration-150 transition" style={{ padding: '1em 2em', borderRadius: '0% 20px', margin: '1em 0em' }}>
          Try Mentra
        </button>
      </Link>
    </section>
  );
};

export default HeroSection;
