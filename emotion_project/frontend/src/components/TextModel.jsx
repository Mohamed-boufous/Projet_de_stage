import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const TextModel = () => {
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [predictions, setPredictions] = useState(null);
  const [topEmotion, setTopEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);

  const handleTextChange = (e) => {
    setText(e.target.value);
    if (error) {
      setError('');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!text.trim()) {
      setError('Please enter some text.');
      return;
    }
    setIsLoading(true);
    setError('');
    setPredictions(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict_text_emotion/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      });
      
      const data = await response.json();

      // Outil de détective : pour voir exactement ce que l'API envoie
      console.log('API Response:', data); 
      
      if (!response.ok) {
        throw new Error(data.detail || 'An error occurred during prediction.');
      }

      // Vérification que les données attendues sont bien là
      if (!data.predictions || !data.predicted_emotion) {
          throw new Error("Invalid data format received from the API.");
      }
      
      setPredictions(data.predictions);
      setTopEmotion(data.predicted_emotion);
      setConfidence(data.confidence);

    } catch (err) {
      if (err instanceof TypeError) {
        setError("API Connection Failed. Is the server running?");
      } else {
        setError(err.message);
      }
      console.error('Error fetching text prediction:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const formatEmotionName = (emotion) => {
    if (!emotion) return '';
    return emotion.charAt(0).toUpperCase() + emotion.slice(1);
  };
  
  const getBarColor = (emotion) => {
    const colors = {
      'joy': '#2ecc71',
      'sadness': '#3498db',
      'anger': '#e74c3c',
      'fear': '#9b59b6',
      'love': '#e84393',
      'surprise': '#f1c40f',
    };
    return colors[emotion.toLowerCase()] || '#bdc3c7';
  };

  return (
    <div className="model-container">
      <header className="app-header">
        <h1>✍️ Text Emotion Analyzer</h1>
        <p>Type a sentence to predict its underlying emotion.</p>
      </header>
      <main>
        <div className="card">
          <form onSubmit={handleSubmit}>
            <textarea
              className="text-input-area"
              value={text}
              onChange={handleTextChange}
              placeholder="e.g., I had a wonderful and happy day!"
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !text.trim()} className="predict-button">
              {isLoading ? <><div className="spinner"></div><span>Analyzing...</span></> : 'Predict Emotion'}
            </button>
          </form>
        </div>
        {error && <div className="card error-card fade-in"><p>❌ {error}</p></div>}
        
        {predictions && topEmotion && (
          <div className="card results-card fade-in">
            <h2>Results</h2>
            <div className="main-result">
              <h3>Primary Emotion</h3>
              <p className="top-emotion" style={{ color: getBarColor(topEmotion) }}>
                {formatEmotionName(topEmotion)}
              </p>
              {confidence && <p>Confidence: {(confidence * 100).toFixed(1)}%</p>}
            </div>
            <div className="predictions-details">
              <h4>All Predictions:</h4>
              <ul className="predictions-list">
                {Object.entries(predictions).sort(([,a],[,b]) => b-a).map(([emotion, probability]) => (
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

export default TextModel;