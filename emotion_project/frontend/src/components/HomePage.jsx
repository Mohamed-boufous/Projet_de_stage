import React from 'react';
import { Link } from 'react-router-dom';
import './../App.css'; // Assurez-vous d'importer le CSS

const HomePage = () => {
  return (
    <div className="home-container">
      <header className="app-header">
        <h1>AI Model Hub 🧠</h1>
        <p>Choose an analysis model to get started</p>
      </header>
      <div className="choice-container">
        <Link to="/image-model" className="choice-card">
          <div className="card-icon">🖼️</div>
          <h2>Image Emotion Analysis</h2>
          <p>Analyze emotions from facial images. Upload a picture to see the result.</p>
        </Link>
        <Link to="/text-model" className="choice-card">
          <div className="card-icon">✍️</div>
          <h2>Text Emotion Analysis</h2>
          <p>Predict emotions from a sentence or a paragraph. Type your text to begin.</p>
        </Link>
      </div>
    </div>
  );
};

export default HomePage;