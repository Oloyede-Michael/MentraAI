import React, { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const CustomRegister = () => {
  const [firstName, setFirstName] = useState('');
  const [nickname, setNickname] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [errorMessage, setErrorMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const togglePasswordVisibility = () => setShowPassword(!showPassword);
  const toggleConfirmPasswordVisibility = () => setShowConfirmPassword(!showConfirmPassword);

  const validatePassword = (password) => {
    const errors = {};
    if (password.length < 8) {
      errors.length = 'Password must be at least 8 characters';
    }
    if (!/[A-Z]/.test(password)) {
      errors.uppercase = 'At least one uppercase letter is required';
    }
    if (!/[a-z]/.test(password)) {
      errors.lowercase = 'At least one lowercase letter is required';
    }
    if (!/[0-9]/.test(password)) {
      errors.number = 'At least one digit is required';
    }
    if (!/[^A-Za-z0-9]/.test(password)) {
      errors.special = 'At least one special character is required';
    }
    return errors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setLoading(true);

    const validationErrors = validatePassword(password);

    if (password !== confirmPassword) {
      validationErrors.confirm = 'Passwords do not match';
    }

    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      setLoading(false);
      return;
    }

    setErrors({});
    const email = localStorage.getItem('mentra_email');

    if (!email) {
      setErrorMessage("Email not found. Please restart the registration process.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch("http://localhost:1000/api/users/customs-register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, firstName, nickname, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        setErrorMessage(data.message || "Something went wrong. Please try again.");
      } else {
        localStorage.removeItem("mentra_email");
        navigate("/login");
      }
    } catch (err) {
      setErrorMessage("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col p-4">
      {/* Brand */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Mentra</h1>
        <div className="w-12 h-1 bg-indigo-600 rounded-full"></div>
      </div>

      {/* Form */}
      <div className="flex flex-1 items-center justify-center">
        <div className="w-full max-w-xl bg-white shadow-xl rounded-2xl p-8">
          <h2 className="text-2xl font-semibold text-gray-900 text-center mb-8">
            Create Your Mentra Profile
          </h2>

          {errorMessage && (
            <p className="text-red-600 text-sm text-center mb-4">{errorMessage}</p>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* First Name */}
            <div>
              <label htmlFor="firstName" className="block text-sm font-medium text-indigo-600 mb-1">
                Full Name
              </label>
              <input
                type="text"
                id="firstName"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                className="w-full px-4 py-3 border-2 border-indigo-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
                placeholder="e.g. John Doe"
                required
              />
            </div>

            {/* Nickname */}
            <div>
              <label htmlFor="nickname" className="block text-sm font-medium text-indigo-600 mb-1">
                Nickname (What you'd like to be called)
              </label>
              <input
                type="text"
                id="nickname"
                value={nickname}
                onChange={(e) => setNickname(e.target.value)}
                className="w-full px-4 py-3 border-2 border-indigo-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
                placeholder="e.g. Doeis"
                required
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-indigo-600 mb-1">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 pr-12 border-2 border-indigo-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
                  placeholder="Enter a strong password"
                  required
                />
                <button
                  type="button"
                  onClick={togglePasswordVisibility}
                  className="absolute top-3 right-3 text-gray-600"
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
              {Object.values(errors).map(
                (err, idx) =>
                  err && (
                    <p key={idx} className="text-red-600 text-xs mt-1">
                      • {err}
                    </p>
                  )
              )}
            </div>

            {/* Confirm Password with toggle */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-indigo-600 mb-1">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  id="confirmPassword"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full px-4 py-3 pr-12 border-2 border-indigo-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
                  placeholder="Re-enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={toggleConfirmPasswordVisibility}
                  className="absolute top-3 right-3 text-gray-600"
                >
                  {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
              {errors.confirm && (
                <p className="text-red-600 text-xs mt-1">• {errors.confirm}</p>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-xl"
            >
              {loading ? "Creating..." : "Create Account"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default CustomRegister;