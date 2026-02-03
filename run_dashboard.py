#!/usr/bin/env python3
"""
Smart News Summarizer & Analyzer - Dashboard Launcher

This script launches the Streamlit dashboard for the Smart News Summarizer & Analyzer.
Make sure you have run the data processing pipeline first to generate the required data files.

Usage:
    python run_dashboard.py

Requirements:
    - streamlit
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - plotly
    - wordcloud
"""

import subprocess
import sys
import os


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib',
        'seaborn', 'plotly', 'wordcloud'
    ]
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    return True


def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'data/processed/articles_with_topics.csv',
        'data/processed/topic_modeling_metadata.json'
    ]
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Please run the data processing pipeline first:")
        print("   1. Run the notebooks/data_verification.ipynb notebook")
        print("   2. Complete all phases (1-6)")
        print("   3. Then run this dashboard")
        return False
    return True


def main():
    """Main launcher function."""
    print("🚀 Starting Smart News Summarizer & Analyzer Dashboard...")
    print("=" * 60)
    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All required packages are installed")
    # Check data files
    print("\n📁 Checking data files...")
    if not check_data_files():
        sys.exit(1)
    print("✅ All required data files found")
    # Launch Streamlit
    print("\n🌐 Launching Streamlit dashboard...")
    print("📱 The dashboard will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    try:
        # Run Streamlit
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                "src/dashboard.py",
                "--server.port", "8501",
                "--server.address", "localhost",
                "--browser.gatherUsageStats", "false"
            ],
            check=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
