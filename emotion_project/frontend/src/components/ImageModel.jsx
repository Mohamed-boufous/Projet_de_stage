import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const ImageModel = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  // Corrigé pour correspondre à la réponse de l'API
  const [confidences, setConfidences] = useState(null); 
  const [topEmotion, setTopEmotion] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError('');
      // Corrigé pour correspondre à la réponse de l'API
      setConfidences(null); 
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => { event.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = (event) => { event.preventDefault(); setIsDragOver(false); };
  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragOver(false);
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      const fileInput = document.getElementById('file-input');
      if (fileInput) fileInput.files = dataTransfer.files;
      handleFileChange({ target: { files: dataTransfer.files } });
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please select an image before predicting.');
      return;
    }
    setIsLoading(true);
    setError('');

    // CORRECTION 1: URL de l'endpoint mise à jour
    const apiUrl = 'http://127.0.0.1:8000/predict_image_emotion/';
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(apiUrl, { method: 'POST', body: formData });
      
      if (!response.ok) {
        let errorDetail = `Error ${response.status}: ${response.statusText}`;
        try {
          const errorJson = await response.json();
          errorDetail = errorJson.detail || errorDetail;
        } catch {}
        throw new Error(errorDetail);
      }

      const data = await response.json();
      
      // CORRECTION 2: Utilisation des bonnes clés de la réponse JSON
      setConfidences(data.confidences); 
      setTopEmotion(data.predicted_emotion);

    } catch (err) {
      if (err instanceof TypeError) {
        setError("API Connection Failed. Is the server running at http://127.0.0.1:8000?");
      } else {
        setError(err.message);
      }
      console.error('Error fetching image prediction:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const formatEmotionName = (emotion) => {
    return emotion.charAt(0).toUpperCase() + emotion.slice(1);
  };

  const getBarColor = (emotion) => {
    const colors = {
      'happy': '#2ecc71', 'surprise': '#f1c40f', 'neutral': '#bdc3c7',
      'sad': '#3498db', 'angry': '#e74c3c', 'fear': '#9b59b6', 'disgust': '#16a085'
    };
    return colors[emotion] || '#3498db';
  };

  return (
    <div className="model-container">
      <header className="app-header">
        <h1>🖼️ Image Emotion Analyzer</h1>
        <p>Upload a facial image to discover the emotion.</p>
      </header>
      <main>
        <div className="card">
          <form onSubmit={handleSubmit}>
            <label
              htmlFor="file-input"
              className={`dropzone ${isDragOver ? 'drag-over' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
              />
              {preview ? (
                <img src={preview} alt="Preview" className="image-preview" />
              ) : (
                <div className="dropzone-prompt">
                  <svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                  <p>Drag and drop an image here, or click to select</p>
                </div>
              )}
            </label>
            <button type="submit" disabled={isLoading || !selectedFile} className="predict-button">
              {isLoading ? (
                <>
                  <div className="spinner"></div>
                  <span>Analyzing...</span>
                </>
              ) : 'Predict Emotion'}
            </button>
          </form>
        </div>
        {error && <div className="card error-card fade-in"><p>❌ {error}</p></div>}
        {/* Corrigé pour correspondre à la réponse de l'API */}
        {confidences && (
          <div className="card results-card fade-in">
            <h2>Results</h2>
            <div className="main-result">
              <h3>Primary Emotion</h3>
              <p className="top-emotion">{formatEmotionName(topEmotion)}</p>
            </div>
            <div className="predictions-details">
              <h4>Prediction Details:</h4>
              <ul className="predictions-list">
                {Object.entries(confidences).sort(([,a],[,b]) => b-a).map(([emotion, probability]) => (
                  <li key={emotion} className="prediction-item">
                    <div className="emotion-info">
                      <span className="emotion-label">{formatEmotionName(emotion)}</span>
                      <span className="emotion-proba">{(probability * 100).toFixed(2)}%</span>
                    </div>
                    <div className="progress-bar-container">
                      <div className="progress-bar" style={{ width: `${probability * 100}%`, backgroundColor: getBarColor(emotion) }}></div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </main>
      <div className="back-link-container">
        <Link to="/" className="back-link">← Back to Home</Link>
      </div>
    </div>
  );
};

export default ImageModel;