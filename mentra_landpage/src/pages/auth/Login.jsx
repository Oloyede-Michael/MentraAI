import React, { useState } from 'react';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg('');

    try {
      const res = await fetch('http://localhost:1000/api/users/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Required for session cookies
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrorMsg(data.message || 'Login failed');
        setLoading(false);
        return;
      }

      // Redirect to dashboard on port 3000
      window.location.href = 'http://localhost:3000';
    } catch (err) {
      console.error('Login error:', err);
      setErrorMsg('Network error. Please try again.');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col p-4">
      {/* Brand at top-left */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Mentra</h1>
        <div className="w-12 h-1 bg-indigo-600 rounded-full"></div>
      </div>

      {/* Main Form Section Centered Vertically */}
      <div className="flex flex-1 items-center justify-center">
        <div className="w-full max-w-xl">
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-semibold text-gray-900 text-center mb-8">
              Log in to your account
            </h2>

            {errorMsg && (
              <p className="text-red-600 text-sm mb-4 text-center">{errorMsg}</p>
            )}

            <form onSubmit={handleLogin}>
              {/* Email Input */}
              <div className="mb-6">
                <label htmlFor="email" className="block text-sm font-medium text-indigo-600 mb-2">
                  Email address
                </label>
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-indigo-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all duration-200 text-gray-900 placeholder-gray-400"
                  placeholder="Enter your email"
                  required
                />
              </div>

              {/* Password Input */}
              <div className="mb-6">
                <label htmlFor="password" className="block text-sm font-medium text-indigo-600 mb-2">
                  Password
                </label>
                <input
                  type="password"
                  id="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-indigo-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all duration-200 text-gray-900 placeholder-gray-400"
                  placeholder="Enter your password"
                  required
                />
              </div>

              {/* Login Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-xl mb-6"
              >
                {loading ? 'Logging in...' : 'Log in'}
              </button>
            </form>

            {/* Register Link */}
            <p className="text-center text-gray-600 mb-6">
              Don't have an account?{' '}
              <a href="/register" className="text-indigo-600 hover:text-indigo-700 font-medium hover:underline transition-colors">
                Sign up
              </a>
            </p>

            {/* Divider */}
            <div className="relative mb-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-gray-500 font-medium">OR</span>
              </div>
            </div>

            {/* Google Login (Optional) */}
            <div className="space-y-3">
              <button className="w-full flex items-center justify-center gap-3 px-4 py-3 border-2 border-gray-200 rounded-xl hover:border-gray-300 hover:bg-gray-50 transition-all duration-200 group">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                </svg>
                <span className="text-gray-700 font-medium">Continue with Google</span>
              </button>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center mt-8 text-sm text-gray-500">
            <p>By logging in, you agree to our Terms of Service and Privacy Policy</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;