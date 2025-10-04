# Cattle vs Buffalo Classifier

A single-page web application that uses TensorFlow.js and MobileNet to classify images between Cattle and Buffalo. The application runs entirely in the browser with no backend required.

## Features

- **Drag & Drop Upload**: Easy image upload with drag and drop functionality
- **Image Preview**: Preview uploaded images before classification
- **Real-time Classification**: Uses TensorFlow.js MobileNet model for instant classification
- **Confidence Scoring**: Shows confidence percentage with visual progress bar
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Clean, modern interface with smooth animations
- **No Backend Required**: All processing happens in the browser

## How It Works

1. **Model Loading**: The application loads a pre-trained MobileNet model from TensorFlow.js
2. **Image Processing**: Uploaded images are preprocessed to 224×224 pixels
3. **Classification**: The model analyzes the image and maps results to Cattle vs Buffalo categories
4. **Result Display**: Shows the predicted type and confidence score with visual feedback

## File Structure

```
├── index.html      # Main HTML structure
├── style.css       # Modern CSS styling
├── script.js       # JavaScript logic and TensorFlow.js integration
└── README.md       # This file
```

## Usage

1. **Open the Application**: Open `index.html` in a modern web browser
2. **Upload Image**: 
   - Drag and drop an image onto the upload area, or
   - Click "Browse Files" to select an image
3. **Preview**: The uploaded image will be displayed in the preview section
4. **Classify**: Click the "Classify Image" button to run the classification
5. **View Results**: The predicted type (Cattle/Buffalo) and confidence score will be displayed

## Technical Details

### Model Information
- **Base Model**: MobileNet V2 (pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Classification**: Maps ImageNet classes to Cattle vs Buffalo categories
- **Processing**: Client-side using TensorFlow.js

### Supported File Types
- JPEG (.jpg, .jpeg)
- PNG (.png)

### Browser Requirements
- Modern browser with JavaScript enabled
- Internet connection (for initial model download)
- Support for ES6+ features

## Classification Logic

The application uses a keyword-based mapping system to classify MobileNet's 1000 ImageNet classes into Cattle vs Buffalo categories:

### Cattle Keywords
- cow, bull, cattle, ox, bovine, heifer, calf

### Buffalo Keywords
- buffalo, bison, water buffalo

### Fallback Logic
If no specific matches are found, the system:
1. Looks for general animal-related classes
2. Makes educated guesses based on class names
3. Provides confidence scores accordingly

## Performance

- **Model Size**: ~14MB (downloaded once and cached)
- **Classification Speed**: ~100-500ms per image
- **Memory Usage**: Minimal (tensors are cleaned up automatically)

## Future Enhancements

- Custom fine-tuned model for better accuracy
- Support for more animal types
- Batch processing for multiple images
- Export results functionality
- Offline mode with cached model

## Credits

- **Powered by**: TensorFlow.js and MobileNet
- **UI Design**: Modern, responsive web design
- **Icons**: Unicode emoji icons

## Troubleshooting

### Common Issues

1. **Model Not Loading**
   - Check internet connection
   - Refresh the page
   - Clear browser cache

2. **Classification Errors**
   - Ensure image is clear and well-lit
   - Try different angles of the animal
   - Check that the image contains a cattle or buffalo

3. **Browser Compatibility**
   - Use Chrome, Firefox, Safari, or Edge
   - Ensure JavaScript is enabled
   - Update to the latest browser version

### Console Logs

The application logs important events to the browser console:
- Model loading status
- Classification results
- Error messages

## License

This project is open source and available under the MIT License.
