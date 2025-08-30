import React, { useEffect, useRef } from 'react';

const FaceResults = ({ results, originalImage }) => {
  const canvasRef = useRef(null);

  const cropFaceFromImage = (imageUrl, bbox, callback) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      const { x, y, width, height } = bbox;
      canvas.width = width;
      canvas.height = height;
      
      ctx.drawImage(img, x, y, width, height, 0, 0, width, height);
      callback(canvas.toDataURL());
    };
    img.src = imageUrl;
  };

  const FaceCard = ({ face, index }) => {
    const [faceImage, setFaceImage] = React.useState(null);

    useEffect(() => {
      if (originalImage && face.bbox) {
        cropFaceFromImage(originalImage, face.bbox, setFaceImage);
      }
    }, [originalImage, face.bbox]);

    const formatConfidence = (confidence) => {
      return (confidence * 100).toFixed(1);
    };

    return (
      <div className="col-md-4 mb-4">
        <div className="card face-card h-100">
          <div className="card-header bg-primary text-white">
            <h6 className="mb-0">
              <i className="bi bi-person-circle me-2"></i>
              Face {index + 1}
            </h6>
          </div>
          {faceImage && (
            <img 
              src={faceImage} 
              className="card-img-top face-bbox" 
              alt={`Detected face ${index + 1}`}
              style={{ maxHeight: '200px', objectFit: 'cover' }}
            />
          )}
          <div className="card-body">
            <div className="row text-center">
              <div className="col-6">
                <div className="border-end">
                  <h5 className="text-primary mb-1">
                    <i className="bi bi-calendar3 me-1"></i>
                    {Math.round(face.age)}
                  </h5>
                  <small className="text-muted">Age</small>
                  {face.age_confidence && (
                    <div className="mt-1">
                      <small className="text-success">
                        {formatConfidence(face.age_confidence)}% confident
                      </small>
                    </div>
                  )}
                </div>
              </div>
              <div className="col-6">
                <h5 className="text-success mb-1">
                  <i className={`bi ${face.gender === 'Male' ? 'bi-person' : 'bi-person-dress'} me-1`}></i>
                  {face.gender}
                </h5>
                <small className="text-muted">Gender</small>
                {face.gender_confidence && (
                  <div className="mt-1">
                    <small className="text-success">
                      {formatConfidence(face.gender_confidence)}% confident
                    </small>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (!results || !results.faces || results.faces.length === 0) {
    return (
      <div className="alert alert-info mt-4" role="alert">
        <i className="bi bi-info-circle me-2"></i>
        No faces detected in the uploaded image.
      </div>
    );
  }

  return (
    <div className="mt-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h4 className="text-primary mb-0">
          <i className="bi bi-eye me-2"></i>
          Detection Results
        </h4>
        <span className="badge bg-primary fs-6">
          {results.faces.length} face{results.faces.length !== 1 ? 's' : ''} detected
        </span>
      </div>
      
      <div className="row">
        {results.faces.map((face, index) => (
          <FaceCard key={index} face={face} index={index} />
        ))}
      </div>

      {results.processing_time && (
        <div className="mt-3 text-center">
          <small className="text-muted">
            <i className="bi bi-clock me-1"></i>
            Processed in {results.processing_time.toFixed(3)}s
          </small>
        </div>
      )}
    </div>
  );
};

export default FaceResults;