import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function EmailVerify() {
  const inputRefs = Array.from({ length: 6 }, () => useRef(null));
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Move focus logic
  const handleChange = (e, idx) => {
    const val = e.target.value;
    if (/^[0-9]{1}$/.test(val)) {
      if (idx < 5) {
        inputRefs[idx + 1].current.focus();
      }
    } else if (val === "") {
      if (idx > 0) {
        inputRefs[idx - 1].current.focus();
      }
    } else {
      e.target.value = "";
    }
  };

  // Handle paste
  const handlePaste = (e) => {
    const paste = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    if (paste.length > 0) {
      paste.split("").forEach((num, idx) => {
        if (inputRefs[idx]?.current) {
          inputRefs[idx].current.value = num;
        }
      });
      if (inputRefs[Math.min(5, paste.length - 1)]) {
        inputRefs[Math.min(5, paste.length - 1)].current.focus();
      }
      e.preventDefault();
    }
  };

  // Handle verify
  const handleVerify = async () => {
    setError('');
    setLoading(true);

    const code = inputRefs.map(ref => ref.current.value).join('');
    const email = localStorage.getItem("mentra_email");

    if (!email || code.length !== 6) {
      setError("Please enter the full 6-digit code.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch("http://localhost:1000/api/users/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, code }),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.message || "Verification failed. Try again.");
      } else {
        navigate("/customs-register");
      }
    } catch (err) {
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col justify-between items-center p-4">
      <div className="w-full max-w-md mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Mentra</h1>
        <div className="w-12 h-1 bg-indigo-600 rounded-full mt-1"></div>
      </div>

      <div className="w-full max-w-md bg-white shadow-md rounded-lg px-6 py-8">
        <h2 className="text-center text-2xl font-semibold text-[#232533] mb-3">
          Confirm Email Address
        </h2>
        <p className="text-center text-[#393a3f] mb-6">
          To verify your email, please enter the six-digit code we sent to your email
        </p>

        {error && (
          <p className="text-red-600 text-sm text-center mb-4">{error}</p>
        )}

        <form className="flex justify-center space-x-2 mb-4" autoComplete="off" onSubmit={(e) => e.preventDefault()}>
          {[0, 1, 2, 3, 4, 5].map((idx) => (
            <input
              key={idx}
              type="text"
              inputMode="numeric"
              maxLength={1}
              ref={inputRefs[idx]}
              className="w-14 h-16 text-2xl text-center bg-[#f7f7fa] border border-[#ededf0] rounded transition focus:outline-none focus:ring-2 focus:ring-[#4156c5]"
              onChange={(e) => handleChange(e, idx)}
              onPaste={handlePaste}
              pattern="[0-9]*"
              autoFocus={idx === 0}
              aria-label={`Digit ${idx + 1}`}
            />
          ))}
        </form>

        <button
          type="submit"
          onClick={handleVerify}
          className="w-full bg-[#4156c5] hover:bg-[#3243a8] text-white font-bold py-2 rounded transition"
          disabled={loading}
        >
          {loading ? "Verifying..." : "Verify"}
        </button>
      </div>

      <footer className="mt-12 text-sm text-gray-500 text-center">
        &copy; Copyright by Alpha Builders 2025
      </footer>
    </div>
  );
}