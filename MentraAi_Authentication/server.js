const dotenv = require("dotenv");
dotenv.config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const session = require("express-session");
const userRoutes = require("./routes/userRoutes");
const errorHandler = require("./middleware/errorHandler");
const cookieParser = require("cookie-parser");

const app = express();

//Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());
app.use(cors({
  origin: process.env.BASE_URL,
  credentials: true,
    })); 

app.use(session({
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: false, //Set to true if using https in production
    maxAge: 24 * 60 * 60 * 1000, // 1 day
  },
}));

//mongo connection
const mongoUri = process.env.MONGO_URI;
mongoose
.connect(mongoUri)
.then(() => {
    console.log("Connected to MongoDB");
})
.catch((error) => {
    console.error("Error connecting to MongoDB:, error");
    process.exit(1);
});
//routes
app.use("/api/users", userRoutes);

app.use(errorHandler);

//server connection
const PORT = process.env.PORT || 1000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});