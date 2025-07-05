// src/components/PageWrapper.jsx
import { motion } from "framer-motion";

const PageWrapper = ({ children }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}     // fade in from below
      animate={{ opacity: 1, y: 0 }}       // appear
      exit={{ opacity: 0, y: -20 }}        // fade out upwards
      transition={{ duration: 0.4 }}
    >
      {children}
    </motion.div>
  );
};

export default PageWrapper;
