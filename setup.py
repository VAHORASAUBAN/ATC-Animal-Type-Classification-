#!/usr/bin/env python3
"""
Setup script for Cattle vs Buffalo Classification System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_dataset():
    """Check if dataset exists"""
    dataset_path = Path("cow-and-buffalo.v1i.tensorflow")
    if dataset_path.exists():
        print("✅ Dataset found")
        return True
    else:
        print("❌ Dataset not found. Please ensure 'cow-and-buffalo.v1i.tensorflow' folder exists")
        return False

def train_model():
    """Train the ML model"""
    print("Training ML model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Model trained successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training model: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["models", "static", "templates"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("Cattle vs Buffalo Classification System Setup")
    print("=" * 50)
    
    # Check dataset
    if not check_dataset():
        print("Please ensure the dataset is in the correct location and try again.")
        return False
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return False
    
    # Train model
    if not train_model():
        print("Failed to train model. Please check the dataset and try again.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the system:")
    print("1. Start the Flask backend: python app.py")
    print("2. Open home.html in your web browser")
    print("\nThe system will use the trained ML model for classification.")
    
    return True

if __name__ == "__main__":
    main()
