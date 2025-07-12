const mongoose = require("mongoose");
const bcrypt = require("bcrypt");

const userSchema = new mongoose.Schema({
  firstName: { type: String, required: false },  // Optional during email-only registration
  surname: { type: String, required: false },    // Optional during email-only registration
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
  },
  nickname: { type: String },
  password: { type: String, required: false },   // Optional at registration
  isAdmin: { type: Boolean, default: false },
  isVerified: { type: Boolean, default: false },
  verificationCode: String,
  codeExpiresIn: Date,
});

// ✅ Pre-save hook for hashing password if it exists
userSchema.pre("save", async function (next) {
  if (!this.password || !this.isModified("password")) return next(); // 🛠️ Skip if password is missing or unchanged
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// ✅ Password matching method
userSchema.methods.matchPassword = async function (enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model("User", userSchema);