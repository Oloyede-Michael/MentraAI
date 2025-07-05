import { useLocation, Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import PageWrapper from "./components/PageWrapper";
import NavBar from "./components/NavBar";
import Footer from "./components/Footer/Footer";
import ScrollToTop from "./components/ScrollToTop"; // ✅ Import it here

import Home from "./pages/Home";
import About from "./pages/AboutUs";
import Contact from "./pages/ContactUs";

function App() {
  const location = useLocation();

  return (
    <div style={{ background:' linear-gradient(to right, #c3dafe, #e9d8fd)'}}>
       
      <NavBar />
      
      <ScrollToTop /> {/* ✅ Automatically scroll to top on route change */}
      
      <AnimatePresence mode="wait" initial={false}>
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<PageWrapper><Home /></PageWrapper>} />
          <Route path="/about" element={<PageWrapper><About /></PageWrapper>} />
          <Route path="/contact" element={<PageWrapper><Contact /></PageWrapper>} />
        </Routes>
      </AnimatePresence>

      <Footer />
    </div>
  );
}

export default App;
