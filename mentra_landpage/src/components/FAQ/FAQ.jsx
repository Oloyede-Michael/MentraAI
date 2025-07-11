import { useState } from "react";
import "./FAQ.css";
import { motion, AnimatePresence } from "framer-motion";


const faqData = [
  {
    question: "What is MentraAI?",
    answer:
      "MentraAI is an emotionally intelligent chatbot designed to support your mental and emotional well-being. It uses AI to have empathetic conversations and offers personalized guidance based on how you feel.",
  },
  {
    question: "How does MentraAI know how I feel?",
    answer:
      "MentraAI uses natural language processing and emotion analysis to detect emotional cues from your words. It can recognize emotions like sadness, stress, happiness, anxiety, and more — and adapts its response accordingly.",
  },
  {
    question: "Can I talk to MentraAI about anything?",
    answer:
      "Yes. MentraAI is built to listen without judgment. You can journal your thoughts, express emotions, or simply talk through your day. It’s a safe space for self-reflection and emotional clarity.",
  },
  {
    question: "Is my data private?",
    answer:
      "Absolutely. MentraAI prioritizes user privacy. Your conversations are kept secure and never shared. It only uses your input to improve the interaction in the moment — not to store or analyze you long-term unless you allow it.",
  },
  {
    question: "How is MentraAI different from other chatbots?",
    answer:
      "MentraAI focuses on empathy and emotional intelligence, not just task-completion. It’s designed to understand you emotionally and respond with context-aware compassion something most AI tools overlook.",
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
