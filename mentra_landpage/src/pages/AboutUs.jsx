import React from 'react';
import "../App.css";
import { motion } from 'framer-motion';
import Magnet from '../components/Magnet'
import CardSwap, { Card } from '../components/CardSwap';

const AboutUs = () => {
  return (
    <div className="font-sans overflow-hidden "  style={{ background: 'linear-gradient(to right, #c3dafe, #e9d8fd)' }} >
      {/* === Section 1: About Us === */}
      <section className="flex flex-wrap items-center justify-between  gap-10 min-h-[500px] h-screen w-[98vw]">
        {/* Left Text */}
        <div className="About flex-1 min-w-[300px] " style={{padding:'2rem 0em 0em 2em', marginBottom:'8.25em'}}>
          <h1 className="text-[6rem] md:text-[8rem] leading-none black-future" >About Us</h1>
          <p className="text-[1.4rem] titi leading-[1.7] max-w-[500px]" >
            At Mentra, we believe in a world where emotional well-being is prioritized. 
            Our platform blends empathy and technology to support mental health with innovation, understanding, and care.
          </p>
          
         <Magnet padding={50} disabled={false} magnetStrength={5}>
          <div className=" w-20 h-20 border-2 cursor-pointer rounded-full flex items-center justify-center" style={{marginTop:'2rem'}}>
            <div className="text-4xl text-gray-600 animate-bounce">
              ↓
            </div>
          </div>
        </Magnet>


        </div>

        {/* Right CardSwap */}
        <div className="flex-1 min-w-[300px] translate-x-[100px] translate-y-[30px]" >
          <CardSwap cardDistance={60} verticalDistance={70} delay={4000}>
            <Card className="w-[25em] h-[20em]">
              <div className=' border-2 border-white rounded-4xl bg-white' style={{padding:'5em'}}></div>
              <h3 style={{marginTop:'0.5em'}}>🌱 Our Mission</h3>
              <p>Empower minds through emotional technology.</p>
              
            </Card>
            <Card className="w-[25em] h-[20em]">
               <div className=' border-2 border-white rounded-4xl  bg-white' style={{padding:'5em'}}></div>
              <h3 style={{marginTop:'0.5em'}} >💡 What We Do</h3>
              <p>We provide empathetic tools that listen, learn, and support your emotional needs.</p>
            </Card>
            <Card className="w-[25em] h-[20em]">
               <div className=' border-2 border-white rounded-4xl bg-white' style={{padding:'5em'}}></div>
              <h3 style={{marginTop:'0.5em'}}>🌍 Our Vision</h3>
              <p>A world where mental health care is accessible, intelligent, and compassionate.</p>
            </Card>
          </CardSwap>
        </div>
      </section>

      {/* === Section 2: Meet Our Team === */}
      <div className='titi' >
       <TeamSection />
      </div>
    </div>
  );
};

export default AboutUs;



const TeamCard = ({ image, name, role, description, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      transition={{
        duration: 1,
        ease: "easeOut",
        delay: index * 0.1,
      }}
      viewport={{ once: true, amount: 0.3 }}
      style={{paddingTop:'4rem'}}
      className="group relative w-[300px] max-w-sm h-[440px]  mx-auto cursor-pointer"
    >
      <div className="relative w-full h-full overflow-hidden rounded-lg group-hover:rounded-full transition-all duration-700 ease-in-out">
        <img
          src={image}
          alt={name}
          className="w-full h-full object-cover group-hover:scale-110 group-hover:rotate-12 transition-all duration-700 ease-in-out"
        />

        {/* Hover Overlay */}
        <div
          className="absolute inset-0 flex items-center justify-center bg-opacity-80 text-black text-center opacity-0 group-hover:opacity-100 transition-all duration-500 ease-in-out "
          style={{
            background: "linear-gradient(to right, #c3dafe, #e9d8fd)" , padding:'1.25rem',
          }}
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            whileHover={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.7 }}
            className="text-sm sm:text-base md:text-lg font-medium leading-relaxed"
          >
            <h3 className="text-xl font-bold " style={{margin:'1rem'}}>{role}</h3>
            <p>{description}</p>
          </motion.div>
        </div>
      </div>

      {/* Name & Role below image */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        style={{padding:'1.25rem'}}
        transition={{ duration: 1, delay: 0.3 + index * 0.1 }}
        className=" text-center"
      >
        <h2 className="text-xl font-semibold text-gray-800 transition-colors duration-300">
          {name}
        </h2>
        <h3 className="text-sm text-gray-600 group-hover:text-gray-700 transition-colors duration-300">
          {role}
        </h3>
      </motion.div>
    </motion.div>
  );
};

const TeamSection = () => {
  const teamMembers = [
    {
      image: "/images/toirat.jpg",
      name: "Ibrahim Toirat",
      role: "Team Leader",
      description:
        "Our guiding force, keeping the team focused, aligned, and moving forward with purpose and passion.",
    },
    {
      image: "/images/michael.jpg",
      name: "Oloyede Michael",
      role: "Project Manager & Full-Stack Developer",
      description:
        "With 4 years of experience, Michael balances full-stack development with effective project management to keep us on track.",
    },
    {
      image: "/images/dami.jpg",
      name: "Emmanuel Damilola",
      role: "UI/UX Designer",
      description:
        "With 2 years of experience, Dami designs smooth, user-friendly, and accessible digital experiences.",
    },
    {
      image: "/images/raphael.jpg",
      name: "Eniaiyejuni Raphael",
      role: "AI Engineer & Backend Developer",
      description:
        "With 3 years of experience, Raphael builds smart systems and strong backends powered by AI.",
    },
    {
      image: "/images/wura.jpg",
      name: "Omilabu Wuraola",
      role: "Frontend Developer & Creative Designer",
      description:
        "A frontend wizard with 3 years of experience who brings life to projects through stunning interfaces and interactive animations.",
    },
    {
      image: "/images/gaji.jpg",
      name: "Yaqub Gaji",
      role: "Backend Developer & Pitcher",
      description:
        "A 4-year backend pro with the smoothest pitch—he builds solid systems and presents them with charisma.",
    },
    {
      image: "/images/dani.jpg",
      name: "Abibi Daniella",
      role: "Backend Developer",
      description:
        "Dani delivers stable, efficient backend systems with 2 years of hands-on experience.",
    },
  ];

  return (
    <section className="Team flex flex-col justify-center items-center  bg-white " style={{padding:'15rem 0rem 15rem', margin:'10rem 0rem'}}>
      <motion.h1
        initial={{ opacity: 0, y: -50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        viewport={{ once: true }}
        className="text-4xl sm:text-5xl Team  lg:text-6xl black-future  text-center"
      >
        Meet Our Team
      </motion.h1>

      <div className="flex flex-wrap items-center justify-center   lg:grid-cols-3 gap-10 max-w-7xl mx-auto " style={{padding:'0em 2rem'}}>
        {teamMembers.map((member, index) => (
          <TeamCard
            key={index}
            image={member.image}
            name={member.name}
            role={member.role}
            description={member.description}
            index={index}
          />
        ))}
      </div>
    </section>
  );
};
