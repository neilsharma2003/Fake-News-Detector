#!/usr/bin/env python3
"""
Launcher script for the Streamlit Fake News Detector app
"""

import subprocess
import sys

def main():
    print("🚀 Launching Fake News Detector Streamlit App...")
    print("📱 Opening in your default web browser...")
    print("🔗 The app will be available at: http://localhost:8501")
    print("\n💡 To stop the app, press Ctrl+C in this terminal")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "--version"], 
                      check=True, capture_output=True)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except subprocess.CalledProcessError:
        print("❌ Error: Streamlit is not installed!")
        print("💡 Please install it first:")
        print("   pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
