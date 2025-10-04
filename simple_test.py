#!/usr/bin/env python3
"""
Simple test to verify the dataset and create a basic model
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def test_dataset():
    """Test if we can read the dataset"""
    print("Testing dataset access...")
    
    dataset_path = Path("cow-and-buffalo.v1i.tensorflow")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print("❌ Dataset folder not found")
        return False
    
    # Check CSV files
    train_csv = dataset_path / "train" / "_annotations.csv"
    valid_csv = dataset_path / "valid" / "_annotations.csv"
    test_csv = dataset_path / "test" / "_annotations.csv"
    
    csv_files = [train_csv, valid_csv, test_csv]
    for csv_file in csv_files:
        if csv_file.exists():
            print(f"✅ Found: {csv_file}")
        else:
            print(f"❌ Missing: {csv_file}")
    
    # Try to read one CSV
    try:
        df = pd.read_csv(train_csv)
        print(f"✅ Successfully read training data: {len(df)} rows")
        print(f"   Classes: {df['class'].unique()}")
        return True
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

def create_simple_model():
    """Create a simple model without heavy dependencies"""
    print("\nCreating simple model...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import joblib
        
        # Load data
        dataset_path = Path("cow-and-buffalo.v1i.tensorflow")
        train_csv = dataset_path / "train" / "_annotations.csv"
        
        df = pd.read_csv(train_csv)
        
        # Map classes to animal types
        class_mapping = {
            'White bufflo': 'Buffalo',
            'Asian buffalo': 'Buffalo', 
            'Wild bufflo': 'Buffalo',
            'Brahman cow': 'Cattle',
            'Holstein': 'Cattle',
            'Charolais': 'Cattle'
        }
        
        df['animal_type'] = df['class'].map(class_mapping)
        df = df.dropna(subset=['animal_type'])
        
        print(f"   Mapped {len(df)} samples to animal types")
        print(f"   Distribution: {df['animal_type'].value_counts().to_dict()}")
        
        # Create simple features (just using bounding box info for now)
        features = df[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = df['animal_type'].values
        
        # Train simple model
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test accuracy
        accuracy = model.score(X_test, y_test)
        print(f"   Model accuracy: {accuracy:.3f}")
        
        # Save model
        joblib.dump(model, "simple_model.joblib")
        
        # Save metadata
        metadata = {
            'class_mapping': class_mapping,
            'model_type': 'RandomForestClassifier',
            'accuracy': accuracy
        }
        
        import json
        with open("simple_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Simple model created and saved")
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def main():
    """Main test function"""
    print("Simple Dataset and Model Test")
    print("=" * 40)
    
    # Test dataset
    if not test_dataset():
        print("\n❌ Dataset test failed")
        return False
    
    # Create simple model
    if not create_simple_model():
        print("\n❌ Model creation failed")
        return False
    
    print("\n🎉 All tests passed!")
    print("   - Dataset is accessible")
    print("   - Simple model created")
    print("   - Ready for web integration")
    
    return True

if __name__ == "__main__":
    main()
