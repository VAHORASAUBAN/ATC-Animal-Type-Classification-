#!/usr/bin/env python3
"""
Demo script to show the Cattle vs Buffalo Classification System
"""

import webbrowser
import time
import subprocess
import sys
from pathlib import Path

def check_system():
    """Check if all system components are present"""
    print("🔍 Checking system components...")
    
    # Check HTML files
    html_files = ['home.html', 'upload.html', 'breeds.html', 'info.html']
    for file in html_files:
        if Path(file).exists():
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
    
    # Check JavaScript files
    js_files = ['script.js', 'breed-database.js', 'ml_classifier.js']
    for file in js_files:
        if Path(file).exists():
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
    
    # Check dataset
    if Path("cow-and-buffalo.v1i.tensorflow").exists():
        print("✅ Dataset - Found")
    else:
        print("❌ Dataset - Missing")
    
    # Check CSS
    if Path("style.css").exists():
        print("✅ style.css - Found")
    else:
        print("❌ style.css - Missing")

def start_web_server():
    """Start the web server"""
    print("\n🌐 Starting web server...")
    try:
        # Start HTTP server in background
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        print("✅ Web server started on http://localhost:8000")
        return process
    except Exception as e:
        print(f"❌ Failed to start web server: {e}")
        return None

def open_demo_pages():
    """Open demo pages in browser"""
    print("\n🚀 Opening demo pages...")
    
    pages = [
        ("Home Page", "http://localhost:8000/home.html"),
        ("Upload Page", "http://localhost:8000/upload.html"),
        ("Breeds Page", "http://localhost:8000/breeds.html"),
        ("Test Page", "http://localhost:8000/test_web_system.html")
    ]
    
    for name, url in pages:
        try:
            webbrowser.open(url)
            print(f"✅ Opened {name}")
            time.sleep(1)  # Small delay between opens
        except Exception as e:
            print(f"❌ Failed to open {name}: {e}")

def show_instructions():
    """Show usage instructions"""
    print("\n📋 How to use the system:")
    print("=" * 50)
    print("1. 🖼️  Upload an image of cattle or buffalo")
    print("2. 🔍 Click 'Analyze Image' button")
    print("3. 📊 View the classification results")
    print("4. 📈 Check confidence scores and breed identification")
    print("5. 🔧 View detailed analysis in debug section")
    
    print("\n🎯 Test with different images:")
    print("- Try various cattle breeds (Holstein, Jersey, etc.)")
    print("- Try different buffalo types (Asian, Wild, etc.)")
    print("- Test with different lighting conditions")
    print("- Check how confidence varies with image quality")
    
    print("\n⚙️  System features:")
    print("- Rule-based classification (always works)")
    print("- ML model integration (when packages installed)")
    print("- Breed identification")
    print("- Confidence scoring")
    print("- Detailed analysis breakdown")

def main():
    """Main demo function"""
    print("🐄 Cattle vs Buffalo Classification System Demo")
    print("=" * 60)
    
    # Check system
    check_system()
    
    # Start web server
    server_process = start_web_server()
    if not server_process:
        print("❌ Cannot start demo without web server")
        return
    
    try:
        # Open demo pages
        open_demo_pages()
        
        # Show instructions
        show_instructions()
        
        print("\n🎉 Demo started successfully!")
        print("📱 The system is now running in your browser")
        print("🛑 Press Ctrl+C to stop the demo")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping demo...")
        if server_process:
            server_process.terminate()
        print("✅ Demo stopped")

if __name__ == "__main__":
    main()
