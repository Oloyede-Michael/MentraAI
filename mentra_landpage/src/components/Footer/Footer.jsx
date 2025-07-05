import React from "react";
import "./Footer.css";



const Footer = () => {
  return (
    <div>
      <div className="footer">
        <h1>MENTRA</h1>
        <div className="footter-icons">
          <h2>©2025. All rights reserved.</h2>
          <div className="footer-bottom">
            <div className="footer-social-icons">
                <a href="" target="_blank" rel="noopener noreferrer" ><i className="bx bxl-twitter"></i></a>
                <a href="" target="_blank" rel="noopener noreferrer"><i className="bx bxl-github"></i></a>
                <a href="" target="_blank" rel="noopener noreferrer"><i className="bx bxl-youtube"></i></a>
                <i class='bx bxl-linkedin'></i>
            </div>
         </div>
        </div>
      </div>
    </div>
  );
};

export default Footer;
