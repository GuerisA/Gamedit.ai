import React from "react";
import { Link } from "react-router-dom";

const Dashboard = () => {
  const videos = Array.from({ length: 20 }, (_, i) => `Video ${i + 1}`);

  return (
    <div className="d-flex" style={{ height: "100vh" }}>
      {/* Sidebar */}
      <nav
        className="bg-dark text-light p-3"
        style={{ width: "250px", minHeight: "100vh" }}
      >
        <h4 className="text-light mb-4">Dashboard</h4>
        <ul className="nav flex-column">
          <li className="nav-item mb-3">
            <Link to="/account" className="nav-link text-light">
              Account
            </Link>
          </li>
          <li className="nav-item mb-3">
            <Link to="/upload" className="nav-link text-light">
              Upload
            </Link>
          </li>
          <li className="nav-item mb-3">
            <Link to="/" className="nav-link text-light">
              Videos
            </Link>
          </li>
        </ul>
      </nav>

      {/* Main Content */}
      <div className="flex-grow-1 p-4 bg-secondary">
        <h2 className="text-light">Videos</h2>
        <div
  className="video-grid"
  style={{
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
    gap: "20px",
    padding: "20px",
  }}
>
  {videos.map((video, index) => (
    <div
      key={index}
      className="video-thumbnail text-center"
      style={{
        backgroundColor: "#2c3e50",
        borderRadius: "8px",
        padding: "10px",
        boxShadow: "0px 5px 15px rgba(0, 0, 0, 0.3)",
      }}
    >
      <img
        src={`https://via.placeholder.com/250x150?text=${video}`}
        alt={video}
        className="rounded mb-2"
        style={{
          width: "100%",
          height: "auto",
          borderRadius: "8px",
        }}
      />
      <p className="text-light m-0">{video}</p>
    </div>
  ))}
</div>

         
      </div>
    </div>
  );
};

export default Dashboard;
