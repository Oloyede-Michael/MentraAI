import React from 'react'
import "../App.css"; 

const ContactUs = () => {
  return (
    <section className=" min-h-screen titi text-black  flex flex-col md:flex-row justify-between items-start md:items-center gap-20" style={{ background: 'linear-gradient(to right, #c3dafe, #e9d8fd)' , padding:'1.5rem 3rem'}}>
      {/* Left Side - Heading and Form */}
      <div className="w-full md:w-2/3 " style={{marginBottom:'3rem'}}>
        <h1 className="text-5xl text-black  md:text-6xl font-light leading-tight " style={{marginBottom:'3rem'}}>
          Let's create something <br /> unforgettable
        </h1>

        <form className="space-y-6 w-full max-w-xl">
          <div className="flex flex-col md:flex-row gap-6">
            <input
              type="text"
              placeholder="Name"
              className="bg-transparent border-b border-gray-500 outline-none w-full placeholder-gray-700"
            />
            <input
              type="email"
              placeholder="Email"
              className="bg-transparent border-b border-gray-500 outline-none w-full placeholder-gray-700"
            />
          </div>

          <textarea
            placeholder="Your message..."
            rows="5"
            className="bg-transparent border-b border-gray-500 outline-none w-full placeholder-gray-700 resize-none overflow-hidden"
            onInput={(e) => {
              e.target.style.height = "auto";
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
          ></textarea>

          <button
            type="submit"
            className="border cursor-pointer border-gray-500  text-white hover:bg-indigo-700 bg-indigo-600 text-[1.2em] hover:scale-110 transition duration-300" style={{borderRadius:'0% 20px', boxShadow: '0px 3px 3px rgb(54, 54, 54)', padding:'1.25rem 3.75rem', margin:'1em 0em'}}
          >
            Submit
          </button>
        </form>
      </div>

      {/* Right Side - Contact Info */}
      <div className="w-full md:w-1/3">
        <h2 className="text-xl font-medium mb-4">Get in touch</h2>
        <a
          href="mailto:mentrahealth@gmail.com"
          className="block text-gray-800 mb-2 hover:text-black transition"
        >
          mentrahealth@gmail.com ↗
        </a>
        <p className="text-gray-800">+234 816 892 4256</p>
      </div>
    </section>
    
  )
}

export default ContactUs