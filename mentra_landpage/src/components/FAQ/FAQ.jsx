import { useState } from "react";
import "./FAQ.css";
import { motion, AnimatePresence } from "framer-motion";

const faqData = [
  {
    question: "What is StampNet?",
    answer: "StampNet is a decentralized timestamping platform that allows users to prove document existence at a specific time using blockchain technology.",
  },
  {
    question: "How does blockchain-based timestamping work?",
    answer: "When you upload a document, StampNet generates a cryptographic hash and stores it on the blockchain, ensuring an immutable record of its existence.",
  },
  {
    question: "Is my document stored on the blockchain?",
    answer: "No, only the hash of your document is stored on the blockchain, ensuring privacy while proving authenticity.",
  },
  {
    question: "Can I verify a previously timestamped document?",
    answer: "Yes, you can upload your document to StampNet, and it will check the hash against the blockchain records.",
  },
  {
    question: "Is StampNet free to use?",
    answer: "StampNet may have gas fees associated with blockchain transactions, but basic verification features could be free.",
  },
];

const FAQ = () => {
  const [openIndex, setOpenIndex] = useState(null);

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="FAQ">
      <div className="faq-container">
        <h2 className="faq-title">Frequently Asked Questions</h2>
        <div className="faq-list">
          {faqData.map((faq, index) => (
            <motion.div
              key={index}
              className="faq-item"
              initial={{ opacity: 0, translateX: "-50px" }}
              whileInView={{ opacity: 1, translateX: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <button className="faq-question" onClick={() => toggleFAQ(index)}>
                {faq.question}
                <motion.span
                  className={`faq-icon ${openIndex === index ? "open" : ""}`}
                  animate={{ rotate: openIndex === index ? 45 : 0 }}
                  transition={{ duration: 0.3 }}
                >
                  +
                </motion.span>
              </button>

              <AnimatePresence initial={false}>
                {openIndex === index && (
                  <motion.div
                    className="faq-answer"
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <p>{faq.answer}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FAQ;
