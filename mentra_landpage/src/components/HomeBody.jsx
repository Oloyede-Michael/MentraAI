import React from 'react';
import "../App.css";

const HomeBody = () => {
  return (
    <div className="titi Okay">
      <div className="w-[300px] bg-white/10 backdrop-blur-lg text-black shadow-md rounded-[15px] mb-10 ml-10" style={{ padding:' 1.3em' , margin: '2em 3em 0em', }} >
        <h1 className="text-[50px] font-bold">It's Okay Not To Be Okay</h1>
        <span className="text-[20px] block mt-4">Let your thoughts breathe, we will listen</span>
      </div>

      <div className="flex justify-end">
        <div className="w-[300px] bg-white/15 backdrop-blur-lg text-black shadow-md rounded-[15px]  " style={{padding: '1em', margin: '3em 5em 0em 0em'}}>
          <h1 className="text-[20px] font-medium">
            Empathetic AI chatbot providing 24/7 mental health support with voice interactions and crisis prevention.
          </h1>
        </div>
      </div>
    </div>
  );
};

export default HomeBody;
