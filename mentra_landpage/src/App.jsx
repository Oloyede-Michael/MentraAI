import { useLocation, Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import PageWrapper from "./components/PageWrapper";
import NavBar from "./components/NavBar";
import Footer from "./components/Footer/Footer";
import ScrollToTop from "./components/ScrollToTop"; // ✅ Import it here

import Home from "./pages/Home";
import About from "./pages/AboutUs";
import Contact from "./pages/ContactUs";
import Register from "./pages/auth/Register";
import Login from "./pages/auth/Login";
import EmailVerify from "./pages/auth/EmailVerify";
import CustomRegister from "./pages/auth/CustomRegister";

function App() {
  const location = useLocation();

  // Define routes where NavBar and Footer should be hidden
  const hideNavAndFooter = ["/register", "/login", "/email-verify", "/customs-register"];
  
  // Check if current route should hide NavBar and Footer
  const shouldHideNavAndFooter = hideNavAndFooter.includes(location.pathname);

  return (
    <div style={{ background: 'linear-gradient(to right, #c3dafe, #e9d8fd)' }}>
      
      {/* Conditionally render NavBar */}
      {!shouldHideNavAndFooter && <NavBar />}
      
      <ScrollToTop /> {/* ✅ Automatically scroll to top on route change */}
      
      <AnimatePresence mode="wait" initial={false}>
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<PageWrapper><Home /></PageWrapper>} />
          <Route path="/about" element={<PageWrapper><About /></PageWrapper>} />
          <Route path="/contact" element={<PageWrapper><Contact /></PageWrapper>} />

          {/* Auth Pages */}
          <Route path="/register" element={<Register />} />
          <Route path="/login" element={<Login />} />
          <Route path="/email-verify" element={<EmailVerify />} />
          <Route path="/customs-register" element={<CustomRegister />} />
        </Routes>
      </AnimatePresence>

      {/* Conditionally render Footer */}
      {!shouldHideNavAndFooter && <Footer />}
    </div>
  );
}

export default App;