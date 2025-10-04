#!/usr/bin/env python3
"""
Cattle vs Buffalo Classification Model Training Script
Uses the cow-and-buffalo.v1i.tensorflow dataset for training
"""

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CattleBuffaloClassifier:
    def __init__(self, dataset_path="cow-and-buffalo.v1i.tensorflow"):
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_mapping = {}
        
    def load_dataset(self):
        """Load and preprocess the dataset from CSV files"""
        print("Loading dataset...")
        
        # Load all CSV files
        train_csv = self.dataset_path / "train" / "_annotations.csv"
        valid_csv = self.dataset_path / "valid" / "_annotations.csv"
        test_csv = self.dataset_path / "test" / "_annotations.csv"
        
        # Read CSV files
        train_df = pd.read_csv(train_csv)
        valid_df = pd.read_csv(valid_csv)
        test_df = pd.read_csv(test_csv)
        
        # Combine all data
        all_data = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        
        print(f"Total samples: {len(all_data)}")
        print(f"Classes found: {all_data['class'].unique()}")
        
        # Map classes to animal types
        self.class_mapping = {
            'White bufflo': 'Buffalo',
            'Asian buffalo': 'Buffalo', 
            'Wild bufflo': 'Buffalo',
            'Brahman cow': 'Cattle',
            'Holstein': 'Cattle',
            'Charolais': 'Cattle'
        }
        
        # Add animal type column
        all_data['animal_type'] = all_data['class'].map(self.class_mapping)
        
        # Remove rows with unknown classes
        all_data = all_data.dropna(subset=['animal_type'])
        
        print(f"After mapping: {len(all_data)} samples")
        print(f"Animal type distribution:")
        print(all_data['animal_type'].value_counts())
        
        return all_data
    
    def extract_features(self, image_path, bbox=None):
        """Extract features from an image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Crop to bounding box if provided
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                image = image[ymin:ymax, xmin:xmax]
            
            # Resize image to standard size
            image = cv2.resize(image, (224, 224))
            
            # Extract color features
            features = []
            
            # 1. Color histogram features
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([image], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # 2. HSV color features
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            for i in range(3):  # HSV channels
                hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # 3. Texture features (LBP-like)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Local Binary Pattern approximation
            lbp = self.compute_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [16], [0, 16])
            features.extend(lbp_hist.flatten())
            
            # 4. Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 5. Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.extend([brightness, contrast])
            
            # 6. Shape features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
            else:
                area = 0
                circularity = 0
            
            features.extend([area, circularity])
            
            # 7. Color moments
            for channel in range(3):
                channel_data = image[:, :, channel].flatten()
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                skewness = self.compute_skewness(channel_data)
                features.extend([mean, std, skewness])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def compute_lbp(self, image, radius=1, n_points=8):
        """Compute Local Binary Pattern"""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ''
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if x < rows and y < cols:
                        if image[x, y] >= center:
                            binary_string += '1'
                        else:
                            binary_string += '0'
                    else:
                        binary_string += '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def compute_skewness(self, data):
        """Compute skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def prepare_features(self, df):
        """Extract features from all images"""
        print("Extracting features from images...")
        
        features_list = []
        labels_list = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Processing image {idx}/{len(df)}")
            
            # Construct image path
            image_path = self.dataset_path / "train" / row['filename']
            if not image_path.exists():
                image_path = self.dataset_path / "valid" / row['filename']
            if not image_path.exists():
                image_path = self.dataset_path / "test" / row['filename']
            
            if not image_path.exists():
                print(f"Image not found: {row['filename']}")
                continue
            
            # Extract features
            bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            features = self.extract_features(image_path, bbox)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(row['animal_type'])
                valid_indices.append(idx)
        
        print(f"Successfully processed {len(features_list)} images")
        
        return np.array(features_list), np.array(labels_list)
    
    def train_model(self, X, y):
        """Train the classification model"""
        print("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {score:.4f}")
            print(f"Classification Report for {name}:")
            print(classification_report(y_test, y_pred))
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        print(f"\nBest model: {best_name} with accuracy {best_score:.4f}")
        
        self.model = best_model
        return best_model, best_score
    
    def save_model(self, model_path="cattle_buffalo_model"):
        """Save the trained model and metadata"""
        if self.model is None:
            print("No model to save!")
            return
        
        # Save model
        joblib.dump(self.model, f"{model_path}.joblib")
        
        # Save metadata
        metadata = {
            'class_mapping': self.class_mapping,
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__
        }
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}.joblib")
        print(f"Metadata saved to {model_path}_metadata.json")
    
    def predict(self, image_path, bbox=None):
        """Predict animal type from image"""
        if self.model is None:
            print("Model not trained!")
            return None
        
        features = self.extract_features(image_path, bbox)
        if features is None:
            return None
        
        # Reshape for single prediction
        features = features.reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            'prediction': prediction,
            'confidence': max(probabilities),
            'probabilities': dict(zip(self.model.classes_, probabilities))
        }

def main():
    """Main training function"""
    print("Cattle vs Buffalo Classification Model Training")
    print("=" * 50)
    
    # Initialize classifier
    classifier = CattleBuffaloClassifier()
    
    # Load dataset
    df = classifier.load_dataset()
    
    # Prepare features
    X, y = classifier.prepare_features(df)
    
    if len(X) == 0:
        print("No valid images found!")
        return
    
    # Train model
    model, accuracy = classifier.train_model(X, y)
    
    # Save model
    classifier.save_model()
    
    print(f"\nTraining completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Model saved successfully!")

if __name__ == "__main__":
    main()
