import React from 'react';
import ReactDOM from 'react-dom/client'; // Use this for React 18+
import App from './App';
import 'bootstrap/dist/css/bootstrap.min.css'; // Ensure Bootstrap is included

// Create the root element
const root = ReactDOM.createRoot(document.getElementById('root')); 

// Render the App component
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
