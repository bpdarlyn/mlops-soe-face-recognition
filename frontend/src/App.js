import React, { useState } from 'react';
import axios from 'axios';
import FaceUploader from './components/FaceUploader';
import FaceResults from './components/FaceResults';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);

  const handleImageUpload = async (file) => {
    setLoading(true);
    setError(null);
    setResults(null);
    
    // Store original image for display
    setOriginalImage(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/infer_age_genre', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setResults(response.data);
      } else {
        setError(response.data.message || 'Error analyzing faces');
      }
    } catch (err) {
      console.error('Error uploading image:', err);
      setError('Error connecting to the server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-4">
      <div className="row justify-content-center">
        <div className="col-md-10">
          <div className="text-center mb-4">
            <h1 className="display-4 text-primary">Face Analytics</h1>
            <p className="lead text-muted">Upload an image to detect age and gender</p>
          </div>
          
          <FaceUploader onImageUpload={handleImageUpload} loading={loading} />
          
          {error && (
            <div className="alert alert-danger mt-4" role="alert">
              <i className="bi bi-exclamation-triangle me-2"></i>
              {error}
            </div>
          )}
          
          {loading && (
            <div className="loading mt-4">
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Analyzing...</span>
              </div>
              <p className="mt-2 text-muted">Analyzing faces...</p>
            </div>
          )}
          
          {results && originalImage && (
            <FaceResults 
              results={results} 
              originalImage={originalImage}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;