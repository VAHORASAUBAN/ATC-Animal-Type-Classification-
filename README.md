# 🐄 Cattle vs Buffalo Classification System# Cattle vs Buffalo Classifier



A professional AI-powered system for classifying cattle and buffalo breeds using advanced computer vision techniques.A single-page web application that uses TensorFlow.js and MobileNet to classify images between Cattle and Buffalo. The application runs entirely in the browser with no backend required.



## 🚀 Quick Start## Features



### Run the System- **Drag & Drop Upload**: Easy image upload with drag and drop functionality

```bash- **Image Preview**: Preview uploaded images before classification

python demo_system.py- **Real-time Classification**: Uses TensorFlow.js MobileNet model for instant classification

```- **Confidence Scoring**: Shows confidence percentage with visual progress bar

- **Responsive Design**: Works on desktop, tablet, and mobile devices

The system will:- **Modern UI**: Clean, modern interface with smooth animations

1. Start a web server on http://localhost:8000- **No Backend Required**: All processing happens in the browser

2. Open your browser with the classification interface

3. Ready to classify cattle and buffalo images!## How It Works



## 📁 Essential Files1. **Model Loading**: The application loads a pre-trained MobileNet model from TensorFlow.js

2. **Image Processing**: Uploaded images are preprocessed to 224×224 pixels

### Core System3. **Classification**: The model analyzes the image and maps results to Cattle vs Buffalo categories

- **`demo_system.py`** - Main launcher (run this file)4. **Result Display**: Shows the predicted type and confidence score with visual feedback

- **`upload.html`** - Main classification interface

- **`script.js`** - Core application logic## File Structure

- **`local_ml_classifier.js`** - Advanced ML classification engine

- **`breed-database.js`** - Database of 41 Indian bovine breeds```

- **`style.css`** - Professional styling├── index.html      # Main HTML structure

├── style.css       # Modern CSS styling

### User Interface├── script.js       # JavaScript logic and TensorFlow.js integration

- **`home.html`** - Landing page└── README.md       # This file

- **`breeds.html`** - Breed information database```

- **`info.html`** - System information

- **`index.html`** - Entry point (redirects to home)## Usage



### Testing & Debug1. **Open the Application**: Open `index.html` in a modern web browser

- **`test_local_classifier.html`** - Advanced ML classifier testing2. **Upload Image**: 

- **`test_web_system.html`** - Basic system functionality test   - Drag and drop an image onto the upload area, or

   - Click "Browse Files" to select an image

### Configuration3. **Preview**: The uploaded image will be displayed in the preview section

- **`cattle_buffalo_model_metadata.json`** - Model metadata from training4. **Classify**: Click the "Classify Image" button to run the classification

5. **View Results**: The predicted type (Cattle/Buffalo) and confidence score will be displayed

## 🎯 Features

## Technical Details

- **41 Indian Bovine Breeds** supported

- **No Internet Required** - completely offline### Model Information

- **Advanced ML Classification** - 6-layer analysis system- **Base Model**: MobileNet V2 (pre-trained on ImageNet)

- **Real-time Processing** - results in <1 second- **Input Size**: 224×224 pixels

- **Professional Interface** - drag-and-drop upload- **Classification**: Maps ImageNet classes to Cattle vs Buffalo categories

- **Detailed Analysis** - feature extraction breakdown- **Processing**: Client-side using TensorFlow.js



## 🧪 Testing### Supported File Types

- JPEG (.jpg, .jpeg)

1. **Main Interface**: http://localhost:8000/upload.html- PNG (.png)

2. **Advanced Testing**: http://localhost:8000/test_local_classifier.html

3. **Basic Testing**: http://localhost:8000/test_web_system.html### Browser Requirements

- Modern browser with JavaScript enabled

## 📊 Supported Breeds- Internet connection (for initial model download)

- Support for ES6+ features

### Buffalo Breeds (6)

- Bhadawari, Jaffrabadi, Mehsana, Murrah, Nili_Ravi, Surti## Classification Logic



### Cattle Breeds (35)The application uses a keyword-based mapping system to classify MobileNet's 1000 ImageNet classes into Cattle vs Buffalo categories:

- Holstein_Friesian, Jersey, Gir, Sahiwal, Red_Sindhi, Hariana

- And 29 more Indian cattle breeds### Cattle Keywords

- cow, bull, cattle, ox, bovine, heifer, calf

## ⚙️ How It Works

### Buffalo Keywords

1. **Upload Image** - Drag & drop or click to select- buffalo, bison, water buffalo

2. **Feature Extraction** - Advanced computer vision analysis

3. **6-Layer Classification** - Multiple ML techniques### Fallback Logic

4. **Breed Identification** - Specific breed predictionIf no specific matches are found, the system:

5. **Results Display** - Confidence scores and analysis1. Looks for general animal-related classes

2. Makes educated guesses based on class names

## 🔧 Technical Details3. Provides confidence scores accordingly



- **Local Processing** - All classification happens in browser## Performance

- **No APIs** - No server dependencies

- **Advanced CV** - RGB/HSV analysis, texture patterns, edge detection- **Model Size**: ~14MB (downloaded once and cached)

- **Multi-layer Analysis** - Color, brightness, texture, edge, HSV, moments- **Classification Speed**: ~100-500ms per image

- **Memory Usage**: Minimal (tensors are cleaned up automatically)

## 🎉 Ready to Use!

## Future Enhancements

Your system is clean, optimized, and ready for production use!
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
