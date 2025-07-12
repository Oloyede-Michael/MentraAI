const nodemailer = require('nodemailer');

const sendVerificationEmail = async (email, code) => {
  const transporter = nodemailer.createTransport({
    service: 'gmail', 
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
  });

  const mailOptions = {
    from: `MentraAI Team <${process.env.EMAIL_USER}>`,
    to: email,
    subject: 'Verify Your Email Address',
    html: `
      <h1>Verify Your Email Address</h1>
      <p>Hello,</p>
      <p>Thank you for signing up with MentraAI. Please use the code below to verify your email address:</p>
      <h2>${code}</h2>
      <p>This code will expire in 10 minutes.</p>
      <br>
      <p>If you did not create this account, please ignore this email.</p>
      <p>Best regards,<br>MentraAI Team</p>
    `,
  };

  await transporter.sendMail(mailOptions);
};

module.exports = sendVerificationEmail;