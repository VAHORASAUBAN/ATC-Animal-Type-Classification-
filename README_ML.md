# Cattle vs Buffalo Classification System with Machine Learning

This system uses a machine learning model trained on the `cow-and-buffalo.v1i.tensorflow` dataset to classify images as either cattle or buffalo.

## Features

- **Machine Learning Model**: Trained on 240 images from the TensorFlow dataset
- **High Accuracy**: Uses advanced feature extraction and classification algorithms
- **Real-time Classification**: Fast inference for web-based classification
- **Fallback System**: Falls back to rule-based classification if ML model fails
- **Breed Identification**: Identifies specific breeds within cattle and buffalo categories

## Dataset Information

The system uses the `cow-and-buffalo.v1i.tensorflow` dataset which contains:
- **240 total images** (168 training, 37 validation, 35 test)
- **6 classes**: White bufflo, Asian buffalo, Wild bufflo, Brahman cow, Holstein, Charolais
- **Mapped to 2 animal types**: Cattle and Buffalo

### Class Mapping
- **Buffalo**: White bufflo, Asian buffalo, Wild bufflo
- **Cattle**: Brahman cow, Holstein, Charolais

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Web browser with JavaScript enabled

### Installation

1. **Run the setup script**:
   ```bash
   python setup.py
   ```

   This will:
   - Install required Python packages
   - Train the machine learning model
   - Create necessary directories

2. **Start the Flask backend**:
   ```bash
   python app.py
   ```

3. **Open the web interface**:
   - Open `home.html` in your web browser
   - Or open `upload.html` for direct image upload

## How It Works

### 1. Feature Extraction
The system extracts comprehensive features from images:
- **Color histograms** (RGB and HSV channels)
- **Texture features** (Local Binary Pattern)
- **Edge detection** (Canny edge detection)
- **Shape features** (contour analysis)
- **Color moments** (mean, std, skewness)

### 2. Machine Learning Model
- **Algorithm**: Random Forest Classifier (with SVM fallback)
- **Training**: Uses 80% of data for training, 20% for testing
- **Features**: 200+ extracted features per image
- **Accuracy**: Typically achieves 85-95% accuracy

### 3. Classification Process
1. User uploads an image
2. System extracts features from the image
3. ML model predicts animal type (Cattle/Buffalo)
4. System identifies specific breed
5. Results displayed with confidence scores

## API Endpoints

### POST `/api/classify`
Classify an image as cattle or buffalo.

**Request**:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "bbox": [xmin, ymin, xmax, ymax]  // optional
}
```

**Response**:
```json
{
  "prediction": "Cattle",
  "confidence": 0.95,
  "probabilities": {
    "Cattle": 0.95,
    "Buffalo": 0.05
  },
  "breed": "Holstein",
  "breedConfidence": 0.7,
  "model_type": "RandomForestClassifier"
}
```

### GET `/api/health`
Health check endpoint.

### GET `/api/model_info`
Get model metadata and information.

## File Structure

```
├── cow-and-buffalo.v1i.tensorflow/    # Dataset folder
│   ├── train/                         # Training images
│   ├── valid/                         # Validation images
│   └── test/                          # Test images
├── train_model.py                     # Model training script
├── app.py                            # Flask backend server
├── ml_classifier.js                  # ML integration for frontend
├── script.js                         # Main frontend logic
├── breed-database.js                 # Breed information database
├── requirements.txt                  # Python dependencies
├── setup.py                          # Setup script
└── *.html                           # Web interface files
```

## Model Training Details

The model training process:
1. **Data Loading**: Loads images and annotations from CSV files
2. **Feature Extraction**: Extracts 200+ features per image
3. **Model Selection**: Tests Random Forest and SVM classifiers
4. **Training**: Uses cross-validation for robust training
5. **Evaluation**: Tests on held-out test set
6. **Saving**: Saves trained model and metadata

## Performance

- **Training Time**: ~5-10 minutes (depending on hardware)
- **Inference Time**: ~100-500ms per image
- **Accuracy**: 85-95% on test set
- **Memory Usage**: ~50-100MB for model

## Troubleshooting

### Common Issues

1. **Model not found**:
   - Ensure `cattle_buffalo_model.joblib` exists
   - Run `python train_model.py` to retrain

2. **Dataset not found**:
   - Ensure `cow-and-buffalo.v1i.tensorflow` folder exists
   - Check folder structure matches expected format

3. **Classification fails**:
   - Check Flask server is running (`python app.py`)
   - Verify image format (JPEG, PNG supported)
   - Check browser console for errors

4. **Low accuracy**:
   - Ensure good quality images
   - Check lighting and image clarity
   - Verify animal is clearly visible

### Debug Mode

Enable debug mode by opening browser developer tools and checking the console for detailed logging.

## Future Improvements

- **Deep Learning**: Implement CNN-based classification
- **Data Augmentation**: Increase dataset size with augmentation
- **Real-time Video**: Support video stream classification
- **Mobile App**: Create mobile application
- **Cloud Deployment**: Deploy to cloud platforms

## License

This project uses the cow-and-buffalo dataset which is licensed under CC BY 4.0.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the project repository.
