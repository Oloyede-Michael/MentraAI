const express = require("express");
const { body } = require("express-validator");
const router = express.Router();

const User = require("../models/User");
const { protect } = require("../middleware/auth");
const { register, login, logout, setNickname, verifyCode } = require("../controller/userController");

//Register 
router.post("/register", [ body("email").isEmail().withMessage("Invalid email address"), body("password").isLength({ min: 6}).withMessage("Password must be at least 6 characters long"),], register );

//email verification
router.post("/verify", verifyCode);

//Login
router.post("/login", [body("email").isEmail().withMessage("Invalid email address"), body("password").notEmpty().withMessage("Password is required"),], login);

//logout
router.post("/logout", protect, logout);

//setnickname
router.post("/set-nickname", protect, setNickname);

module.exports = router;