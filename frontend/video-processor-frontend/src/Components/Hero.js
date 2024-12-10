import React from 'react';
import { Link } from 'react-router-dom';

const Hero = () => {
  return (
    <div className="hero bg-light text-center p-5">
      <h1>Welcome to Video Processor</h1>
      <p>The ultimate platform for AI-powered video editing!</p>
      <div className="mt-4">
        <Link to="/login" className="btn btn-primary me-2">
          Login
        </Link>
        <Link to="/register" className="btn btn-secondary">
          Sign Up
        </Link>
      </div>
    </div>
  );
};

export default Hero;
