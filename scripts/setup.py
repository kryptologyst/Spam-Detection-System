#!/usr/bin/env python3
"""Setup script for the spam detection system."""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🛡️ Spam Detection System - Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("❌ Failed to install package")
        sys.exit(1)
    
    # Create necessary directories
    print("\nCreating directories...")
    directories = ["data", "models", "logs", "assets", "assets/plots", "assets/models"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Download NLTK data (if available)
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except ImportError:
        print("⚠️ NLTK not available, skipping data download")
    except Exception as e:
        print(f"⚠️ NLTK data download failed: {e}")
    
    # Run basic test
    print("\nRunning basic system test...")
    if not run_command("python scripts/test_system.py", "System test"):
        print("⚠️ Basic test failed, but setup completed")
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python scripts/train.py' to train models")
    print("2. Run 'streamlit run demo/app.py' to launch the demo")
    print("3. Check the README.md for more information")

if __name__ == "__main__":
    main()
