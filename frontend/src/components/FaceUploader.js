import React, { useRef, useState } from 'react';

const FaceUploader = ({ onImageUpload, loading }) => {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file);
    } else {
      alert('Please select a valid image file');
    }
  };

  const handleFileInput = (event) => {
    const file = event.target.files[0];
    handleFileSelect(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div 
      className={`upload-area ${dragOver ? 'drag-over' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
      style={{ cursor: loading ? 'not-allowed' : 'pointer' }}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInput}
        accept="image/*"
        style={{ display: 'none' }}
        disabled={loading}
      />
      
      <div className="text-center">
        <i className="bi bi-cloud-upload display-1 text-primary mb-3"></i>
        <h4 className="mb-2">Upload an Image</h4>
        <p className="text-muted mb-0">
          Drag and drop an image here, or click to select a file
        </p>
        <small className="text-muted">
          Supported formats: JPG, PNG, GIF
        </small>
      </div>
    </div>
  );
};

export default FaceUploader;