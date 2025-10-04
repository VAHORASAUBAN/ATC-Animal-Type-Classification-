#!/usr/bin/env python3
"""
Modified Training Script for Indian Bovine Breeds Dataset
Adapts the original training script to work with folder-based classification
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
import random
warnings.filterwarnings('ignore')

class IndianBovineClassifier:
    def __init__(self, dataset_path=r"C:\Users\SAUBAN VAHORA\Downloads\archive\Indian_bovine_breeds\Indian_bovine_breeds"):
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_mapping = {}
        
        # Define which breeds are cattle vs buffalo
        self.buffalo_breeds = {
            'Bhadawari', 'Jaffrabadi', 'Mehsana', 'Murrah', 'Nili_Ravi', 'Surti'
        }
        
        # All other breeds are cattle
        self.cattle_breeds = {
            'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Brown_Swiss',
            'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 
            'Holstein_Friesian', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod',
            'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda',
            'Nagori', 'Nagpuri', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
            'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Tharparkar', 'Toda',
            'Umblachery', 'Vechur'
        }
        
    def load_dataset(self):
        """Load and preprocess the dataset from folder structure"""
        print("Loading Indian Bovine Breeds dataset...")
        
        data_rows = []
        
        # Get all breed folders
        breed_folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]
        
        print(f"Found {len(breed_folders)} breed folders")
        
        for folder in breed_folders:
            breed_name = folder.name
            
            # Determine if this breed is cattle or buffalo
            if breed_name in self.buffalo_breeds:
                animal_type = 'Buffalo'
            elif breed_name in self.cattle_breeds:
                animal_type = 'Cattle'
            else:
                # Unknown breed - make educated guess based on name
                if any(buffalo_term in breed_name.lower() for buffalo_term in ['murrah', 'buffalo', 'jaffrabadi', 'mehsana', 'surti', 'nili']):
                    animal_type = 'Buffalo'
                else:
                    animal_type = 'Cattle'
                    
                print(f"⚠️  Unknown breed '{breed_name}' classified as {animal_type}")
            
            # Get all images in this folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images = [f for f in folder.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            
            print(f"📁 {breed_name}: {len(images)} images → {animal_type}")
            
            # Add each image to the dataset
            for img_path in images:
                data_rows.append({
                    'filename': str(img_path),
                    'breed': breed_name,
                    'animal_type': animal_type,
                    'folder': breed_name
                })
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        print(f"\n📊 Dataset Summary:")
        print(f"Total images: {len(df)}")
        print(f"Breeds found: {df['breed'].nunique()}")
        print(f"Animal type distribution:")
        print(df['animal_type'].value_counts())
        
        # Create class mapping
        self.class_mapping = {}
        for breed in df['breed'].unique():
            animal_type = df[df['breed'] == breed]['animal_type'].iloc[0]
            self.class_mapping[breed] = animal_type
        
        return df
    
    def extract_features(self, image_path):
        """Extract features from an image (same as original)"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to standard size (no bounding box for folder-based dataset)
            image = cv2.resize(image, (224, 224))
            
            # Extract features (same as original script)
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
        """Compute Local Binary Pattern (same as original)"""
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
        """Compute skewness of data (same as original)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def prepare_features(self, df, max_images_per_breed=100):
        """Extract features from all images with optional sampling"""
        print("Extracting features from images...")
        
        features_list = []
        labels_list = []
        breed_labels_list = []
        
        # Sample images if too many per breed
        sampled_df = []
        for breed in df['breed'].unique():
            breed_data = df[df['breed'] == breed]
            if len(breed_data) > max_images_per_breed:
                breed_data = breed_data.sample(n=max_images_per_breed, random_state=42)
            sampled_df.append(breed_data)
        
        df_sampled = pd.concat(sampled_df, ignore_index=True)
        print(f"Using {len(df_sampled)} images (max {max_images_per_breed} per breed)")
        
        for idx, row in df_sampled.iterrows():
            if idx % 100 == 0:
                print(f"Processing image {idx}/{len(df_sampled)}")
            
            # Extract features
            features = self.extract_features(row['filename'])
            
            if features is not None:
                features_list.append(features)
                labels_list.append(row['animal_type'])
                breed_labels_list.append(row['breed'])
        
        print(f"Successfully processed {len(features_list)} images")
        
        return np.array(features_list), np.array(labels_list), np.array(breed_labels_list)
    
    def train_model(self, X, y):
        """Train the classification model (same as original)"""
        print("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
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
            'model_type': type(self.model).__name__,
            'dataset_type': 'Indian_Bovine_Breeds',
            'total_breeds': len(self.class_mapping),
            'buffalo_breeds': list(self.buffalo_breeds),
            'cattle_breeds': list(self.cattle_breeds)
        }
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}.joblib")
        print(f"Metadata saved to {model_path}_metadata.json")

def main():
    """Main training function"""
    print("🐄 Indian Bovine Breeds Classification Model Training")
    print("=" * 60)
    
    # Initialize classifier
    classifier = IndianBovineClassifier()
    
    # Check if dataset exists
    if not classifier.dataset_path.exists():
        print(f"❌ Dataset not found at: {classifier.dataset_path}")
        print("Please check the dataset path in the script")
        return
    
    # Load dataset
    df = classifier.load_dataset()
    
    # Prepare features (limit images per breed for faster training)
    X, y, breeds = classifier.prepare_features(df, max_images_per_breed=50)
    
    if len(X) == 0:
        print("No valid images found!")
        return
    
    # Train model
    model, accuracy = classifier.train_model(X, y)
    
    # Save model
    classifier.save_model()
    
    print(f"\n🎉 Training completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Model saved successfully!")
    print(f"Dataset: {len(df)} total images, {df['breed'].nunique()} breeds")
    print(f"Training: {len(X)} images used for training")

if __name__ == "__main__":
    main()