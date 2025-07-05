import React, { useEffect, useState } from 'react';


const Body = () => {
  return (
    <div className="w-full">
      {/* Hero Section */}
      <div className="min-h-screen w-full flex flex-col justify-center items-center text-black   text-center titi" style={{padding:'1em'}}>
        <h1 className="text-3xl md:text-4xl font-semibold mb-4">
          Mentra is a gentle companion for when life feels overwhelming
        </h1>
        <h2 className="text-[20px] md:text-2xl" style={{paddingTop:'0.5em'}}>Take the First Step Toward Mental Clarity</h2>
        <div className="" style={{padding:'1.5em 0em'}}>
          <button className="bg-indigo-600 text-white  text-[19px] shadow-2xl hover:scale-110 cursor-pointer transition-transform duration-300" style={{borderRadius: '0% 20px',padding:'1em 2em'}}>
            Check it Out
          </button>
        </div>
      </div>

      {/* How it Works */}
    </div>
  );
};

export default Body;
