const crypto = require("crypto");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
const User = require("../models/User");
const sendVerificationEmail = require("../utils/sendVerificationEmail")
const errorHandler = require("../middleware/errorHandler");
const { validationResult } = require("express-validator");
//const sendResetEmail =

//function for error handling
const handleError = (res, statusCode, message) => {
    res.status(statusCode).json({ message});
};

//Register user
exports.register = async (req, res) => {
    try {
        //hande validation error
        const errors = validationResult(req);
        if(!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }
        
        const { firstName, surname, email, password } = req.body;
        
        const normalizedEmail = email.toLowerCase().trim();
        const existingUser = await User.findOne({ email: normalizedEmail });
        if (existingUser) {
            return handleError(res, 400, "Email already exists");
        }

        const code = Math.floor(100000 + Math.random() * 900000).toString();
        const expiresIn = new Date(Date.now() + 10 * 60 * 1000); //10 minuites

        const newUser = new User({
            firstName,
            surname,
            email: normalizedEmail,
            password,
            verificationCode: code,
            codeExpiresIn: expiresIn,
        });
        await newUser.save();
        const token = jwt.sign({ id: newUser._id }, process.env.JWT_SECRET, { expiresIn: "7d" });

        //send verificationCode 
        await sendVerificationEmail(newUser.email, code,);

        res.status(201).json({
            message:"Reistration successful. please check your email to verify your account", token, nicknameSet: false,
        });
    } catch (error) {
        console.error("Registration error:", error);
        handleError(res, 500, "Internal server error");
    }
};

//Email verification
exports.verifyCode = async (req, res) => {
  try {
    const { email, code } = req.body;
    const user = await User.findOne({ email });

    if (!user) return res.status(404).json({ message: "User not found" });

    if (user.isVerified) return res.status(400).json({ message: "Already verified" });

    if (user.verificationCode !== code || user.codeExpiresIn < new Date()) {
      return res.status(400).json({ message: "Invalid or expired code" });
    }

    user.isVerified = true;
    user.verificationCode = null;
    user.codeExpiresIn = null;
    await user.save();

    res.json({ message: "Email verified successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

//Login
exports.login = async (req, res) => {
    try {
        const { email, password } = req.body;
        const normalizedEmail = email.toLowerCase().trim();

        const user = await User.findOne({ email: normalizedEmail});
            if (!user) {
                return handleError(res, 400, "User not found");
            }
            const isMatch = await user.matchPassword(password);
            if (!isMatch) {
                return handleError(res, 400, "Invalid credentials");
            }
            if (!user.isVerified) {
                return handleError(res, 403, "Please verify your email first")
            }
            req.session.userId = user._id;
            req.session.isAdmin = user.isAdmin || false;
            const nicknameMessage = user.nickname 
            ? `Welcome back, ${user.nickname}!`
            : "Welcome! Please set your nickname.";

            res.status(200).json({
                message: nicknameMessage,
                message: "Login successful",
                user: {
                    id: user._id,
                    firstName: user.firstName,
                    surname: user.surname,
                    email: user.email,
                    isVerified: user.isVerified,
                    nicknameSet: !!user.nickname, // Check if nickname is set
                },
            });
    } catch (error) {
        console.error("Login error:", error);
        handleError(res, 500, "Internal server error");
    }
};

//set nickname
exports.setNickname = async (req, res) => {
  try {
    const { nickname } = req.body;
    const userId = req.session.userId;

    if (!userId) return handleError(res, 401, "Unauthorized");
    if (!nickname || nickname.length < 2)
      return handleError(res, 400, "Nickname must be at least 2 characters");

    const user = await User.findById(userId);
    if (!user) return handleError(res, 404, "User not found");

    user.nickname = nickname;
    await user.save();

    res.status(200).json({
      message: `Nickname set to ${nickname}`,
      nicknameSet: true,
    });
  } catch (error) {
    console.error("Set nickname error:", error);
    handleError(res, 500, "Internal server error");
  }
};

// Complete registration (after email is verified)
exports.completeRegistration = async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { email, firstName, nickname, password } = req.body;
    const user = await User.findOne({ email: email.toLowerCase().trim() });

    if (!user) return handleError(res, 404, "User not found");
    if (!user.isVerified) return handleError(res, 403, "Please verify your email first");
    if (user.password) return handleError(res, 400, "User already completed registration");

    user.firstName = firstName;
    user.nickname = nickname;
    user.password = password;
    await user.save();

    res.status(200).json({
      message: "Registration completed successfully",
      nicknameSet: true,
    });
  } catch (error) {
    console.error("Complete registration error:", error);
    handleError(res, 500, "Internal server error");
  }
};


//Logout 
exports.logout = async (req, res) => {
    try {
        req.session.destroy((err) => {
            if (err) {
                return handleError(res, 500, "Logout Failed");
            }
            res.clearCookie("connect.sid");
            res.status(200).json({ message: "Logged out successfully." })
        });
    } catch (error) {
        console.error("Logout error:", error);
        handleError(res, 500, "Internal server error");
    }
};