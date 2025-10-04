#!/usr/bin/env python3
"""
Dataset Explorer Script
Analyzes the structure of your Indian bovine breeds dataset
"""

import os
from pathlib import Path
import pandas as pd

def explore_dataset(dataset_path):
    """Explore the structure of the dataset"""
    print("🔍 Exploring Indian Bovine Breeds Dataset")
    print("=" * 60)
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    print(f"📁 Dataset location: {dataset_path}")
    print(f"📊 Dataset exists: ✅")
    
    # List all items in the dataset directory
    print("\n📂 Directory contents:")
    items = list(dataset_path.iterdir())
    
    folders = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    print(f"📁 Folders found: {len(folders)}")
    for folder in folders[:10]:  # Show first 10 folders
        print(f"   • {folder.name}")
    if len(folders) > 10:
        print(f"   ... and {len(folders) - 10} more folders")
    
    print(f"\n📄 Files found: {len(files)}")
    for file in files[:10]:  # Show first 10 files
        print(f"   • {file.name}")
    if len(files) > 10:
        print(f"   ... and {len(files) - 10} more files")
    
    # Look for common dataset patterns
    print("\n🔍 Looking for dataset patterns...")
    
    # Check for CSV files
    csv_files = [f for f in files if f.suffix.lower() == '.csv']
    if csv_files:
        print(f"📊 CSV files found: {len(csv_files)}")
        for csv_file in csv_files:
            print(f"   • {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                print(f"     - Rows: {len(df)}, Columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"     - Sample data: {df.iloc[0].to_dict()}")
            except Exception as e:
                print(f"     - Error reading CSV: {e}")
    
    # Check for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in files if f.suffix.lower() in image_extensions]
    print(f"🖼️  Image files in root: {len(image_files)}")
    
    # Check subfolders for images and structure
    total_images = 0
    class_folders = []
    
    for folder in folders:
        folder_images = []
        try:
            for item in folder.iterdir():
                if item.is_file() and item.suffix.lower() in image_extensions:
                    folder_images.append(item)
                    total_images += 1
        except PermissionError:
            print(f"   ⚠️  Permission denied: {folder.name}")
            continue
        
        if folder_images:
            class_folders.append((folder.name, len(folder_images)))
            print(f"   📁 {folder.name}: {len(folder_images)} images")
    
    print(f"\n📈 Summary:")
    print(f"   • Total images found: {total_images}")
    print(f"   • Folders with images: {len(class_folders)}")
    
    if class_folders:
        print(f"   • Possible classes:")
        for class_name, count in class_folders:
            print(f"     - {class_name}: {count} images")
    
    # Suggest dataset format
    print(f"\n💡 Dataset Format Analysis:")
    if csv_files and total_images > 0:
        print("   📊 Appears to be CSV + Images format")
        print("   📝 Recommendation: Check CSV files for annotations")
    elif class_folders:
        print("   📁 Appears to be folder-based classification")
        print("   📝 Recommendation: Use folder names as class labels")
    else:
        print("   ❓ Unknown format - need manual inspection")
    
    return {
        'path': dataset_path,
        'folders': folders,
        'files': files,
        'csv_files': csv_files,
        'image_files': image_files,
        'class_folders': class_folders,
        'total_images': total_images
    }

if __name__ == "__main__":
    # Your dataset path
    dataset_path = r"C:\Users\SAUBAN VAHORA\Downloads\archive\Indian_bovine_breeds\Indian_bovine_breeds"
    
    try:
        result = explore_dataset(dataset_path)
        
        print(f"\n🚀 Next Steps:")
        print("1. Review the dataset structure above")
        print("2. Run the training script adapter based on the format")
        print("3. Start training with your Indian bovine breeds dataset")
        
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")
        print("Please check the dataset path and permissions")