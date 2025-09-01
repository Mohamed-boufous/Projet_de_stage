import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './components/HomePage';
import ImageModel from './components/ImageModel';
import TextModel from './components/TextModel';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-background">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/image-model" element={<ImageModel />} />
          <Route path="/text-model" element={<TextModel />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;