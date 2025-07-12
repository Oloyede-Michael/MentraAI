// import React from "react";
// import Magnet from '../components/Magnet'
// import { Link } from "react-router-dom";

// const NavBar = () => {
//   const navItems = [
//     { label: "AboutUs", path: "/about", type: "link" },
//     { label: "Contact", path: "/contact", type: "link" },
//     { label: "Features", type: "text" }, // not clickable
//     { label: "Docs", type: "text" },     // not clickable
//   ];

//   return (
//       <div
//         className="w-full sticky top-0 z-50" 
//       >
//         <div
//           className="flex justify-between items-center"
//           style={{ padding: "1.5em 2em 0em" }}
//         >
//           {/* Logo */}
//           <div className="text-[30px] backdrop-blur-2xl black-future font-black font-black-future">
//             <Link to="/">Mentra</Link>
//           </div>

//           {/* Navigation Items */}
//           <ul className="flex gap-[25px] titi bg-white/50 backdrop-blur-md rounded-full shadow-xl" style={{ padding: "1em 2em" }}>
//             {navItems.map((item, index) => (
//               <li
//                 key={index}
//                 className="text-base text-black delay-100 text-[18px] hover:text-indigo-600 hover:scale-110 transition"
//               >
//                 {item.type === "link" ? (
//                   <Link to={item.path}>{item.label}</Link>
//                 ) : (
//                   <span className="cursor-default hover:text-indigo-600 text-black ">{item.label}</span>
//                 )}
//               </li>
//             ))}
//           </ul>

//           {/* Button */}
//           <Magnet padding={50} disabled={false} magnetStrength={5}>
//             <Link to="/dashboard">
//               <button
//                 className="bg-indigo-600 cursor-pointer text-[18px] titi text-white shadow-2xl hover:bg-indigo-800 transition"
//                 style={{ padding: "1em 2em", borderRadius: "0% 20px" }}
//               >
//                 Launch App
//               </button>
//             </Link>
//           </Magnet>
//         </div>
//       </div>
//   );
// };

// export default NavBar;


import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import Magnet from "../components/Magnet";

const NavBar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const navItems = [
    { label: "AboutUs", path: "/about", type: "link" },
    { label: "Contact", path: "/contact", type: "link" },
    { label: "Features", type: "text" },
    { label: "Resources", type: "text" },
  ];

  useEffect(() => {
    document.body.style.overflow = isOpen ? "hidden" : "";
    return () => (document.body.style.overflow = "");
  }, [isOpen]);

  const closeMenu = () => setIsOpen(false);

  return (
    <div className="w-full sticky top-0 z-50">
      <div className="flex justify-between items-center" style={{ padding: "1.5rem 2rem 0rem" }}>
        {/* Logo */}
        <div className="text-[30px] black-future font-black font-black-future bg-transparent backdrop-blur-2xl">
          <Link to="/">Mentra</Link>
        </div>

        {/* Desktop Nav Items */}
        <ul className="hidden md:flex gap-[25px] titi bg-white/50 backdrop-blur-md rounded-full shadow-xl" style={{ padding: "1rem 2rem" }}>
          {navItems.map((item, index) => (
            <li
              key={index}
              className="text-base text-black delay-100 text-[18px] hover:text-indigo-600 hover:scale-110 transition"
            >
              {item.type === "link" ? (
                <Link to={item.path}>{item.label}</Link>
              ) : (
                <span className="cursor-default text-black hover:text-indigo-600">{item.label}</span>
              )}
            </li>
          ))}
        </ul>

        {/* Desktop Button */}
        <div className="hidden md:block">
          <Magnet padding={50} disabled={false} magnetStrength={5}>
            <Link to="/register">
              <button
                className="bg-indigo-600 cursor-pointer text-[18px] titi text-white shadow-2xl hover:bg-indigo-800 transition"
                style={{ padding: "1rem 2rem", borderRadius: "0% 20%" }}
              >
                Launch App
              </button>
            </Link>
          </Magnet>
        </div>

        {/* Mobile Hamburger Button */}
        <div className="md:hidden">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="text-3xl text-black cursor-pointer focus:outline-none"
          >
            {isOpen ? "✕" : "☰"}
          </button>
        </div>
      </div>

      {/* Mobile Full-Screen Menu */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Overlay */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.5 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="fixed inset-0 bg-black z-40"
              onClick={closeMenu}
            />

            {/* Menu Panel */}
            <motion.div
              initial={{ opacity: 0, y: -30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              transition={{ duration: 0.3 }}
              className="fixed inset-0 backdrop-blur-2xl z-50 bg-transparent flex flex-col"
              style={{ padding: "2rem 2.5rem" }}
            >
              {/* Mobile Top Bar */}
              <div className="flex text-white justify-between items-center" style={{ marginBottom: "2rem" }}>
                {/* Logo */}
                <div className="text-[28px] bg-transparent backdrop-blur-3xl black-future font-black">
                  <Link to="/" onClick={closeMenu}>Mentra</Link>
                </div>

                {/* Close Icon */}
                <button
                  onClick={closeMenu}
                  className="text-3xl text-white cursor-pointer focus:outline-none"
                >
                  ✕
                </button>
              </div>

              {/* Nav Items */}
              <div className="flex flex-col gap-10" style={{ padding: "1em" }}>
                {navItems.map((item, index) => (
                  <div
                    key={index}
                    className="text-[30px] hover:translate-x-3.5 cursor-pointer text-white titi hover:text-indigo-600 transition"
                    onClick={closeMenu}
                  >
                    {item.type === "link" ? (
                      <Link to={item.path}>{item.label}</Link>
                    ) : (
                      <span className="cursor-default">{item.label}</span>
                    )}
                  </div>
                ))}

                {/* Launch App Button */}
                <div className="" style={{ paddingTop: "1rem" }}>
                  <Link to="/register" onClick={closeMenu}>
                    <button className="bg-indigo-700 text-white w-[10rem] text-[18px] rounded-2xl hover:bg-indigo-700 hover:scale-105 duration-100 cursor-pointer transition" style={{ padding: "1rem" }}>
                      Launch App
                    </button>
                  </Link>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default NavBar;