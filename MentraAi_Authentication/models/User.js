const mongoose = require("mongoose");
const bcrypt = require("bcrypt");

const userSchema = new mongoose.Schema({
    firstName: { type: String, required: true,},
    surname: { type: String, required: true },
    email: { type: String, required: true, unique: true, lowercase: true, trim: true, },
    nickname: { type: String, },
    password: { type: String, required: true },
    isAdmin: { type: Boolean, default: false },
    isVerified: { type: Boolean, default: false },
    verificationCode: String,
    codeExpiresIn: Date,
});

//hash password
userSchema.pre("save", async function (next) {
    if (this.isModified("password")) return next();

    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
});

//match password
userSchema.methods.matchPassword = async function (enteredPassword) {
    return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model("User", userSchema);