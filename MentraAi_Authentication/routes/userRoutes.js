const express = require("express");
const { body } = require("express-validator");
const router = express.Router();

const User = require("../models/User");
const { protect } = require("../middleware/auth");
const {
  register,
  completeRegistration,
  login,
  logout,
  setNickname,
  verifyCode,
} = require("../controller/userController");

// Step 1: Register with email only
router.post(
  "/register",
  [
    body("email").isEmail().withMessage("Invalid email address"),
    body("password")
      .optional() // Password is optional in step 1
      .isLength({ min: 6 })
      .withMessage("Password must be at least 6 characters long"),
  ],
  register
);

// Step 2: Email verification
router.post("/verify", verifyCode);

// Step 3: Complete registration (firstName, nickname, password)
router.post(
  "/customs-register",
  [
    body("email").isEmail().withMessage("Invalid email"),
    body("firstName").notEmpty().withMessage("First name is required"),
    body("nickname")
      .isLength({ min: 2 })
      .withMessage("Nickname must be at least 2 characters"),
    body("password")
      .isLength({ min: 8 }).withMessage("Password must be at least 8 characters long")
      .matches(/[A-Z]/).withMessage("Password must include an uppercase letter")
      .matches(/[a-z]/).withMessage("Password must include a lowercase letter")
      .matches(/[0-9]/).withMessage("Password must include a digit")
      .matches(/[^A-Za-z0-9]/).withMessage("Password must include a special character"),
  ],
  completeRegistration
);

// Login
router.post(
  "/login",
  [
    body("email").isEmail().withMessage("Invalid email address"),
    body("password").notEmpty().withMessage("Password is required"),
  ],
  login
);

// Logout
router.post("/logout", protect, logout);

// Set nickname
router.post("/set-nickname", protect, setNickname);

module.exports = router;