"""
Test script for the Streamlit dashboard.
This script tests the data loading functionality without running the full Streamlit app.
"""

import json
import os
import sys

import pandas as pd

# Add src to path
sys.path.append("src")


def test_data_loading():
    """Test if the dashboard can load required data files."""
    print("🧪 Testing Dashboard Data Loading...")
    print("=" * 50)

    # Test data file paths
    articles_path = "data/processed/articles_with_topics.csv"
    metadata_path = "data/processed/topic_modeling_metadata.json"

    # Check if files exist
    print("📁 Checking data files...")
    print(
        f"   Articles: {articles_path} - "
        f"{'✅' if os.path.exists(articles_path) else '❌'}"
    )
    print(
        f"   Metadata: {metadata_path} - "
        f"{'✅' if os.path.exists(metadata_path) else '❌'}"
    )

    if not os.path.exists(articles_path) or not os.path.exists(metadata_path):
        print("\n❌ Required data files not found!")
        print("💡 Please run the data processing pipeline first:")
        print("   1. Run notebooks/data_verification.ipynb")
        print("   2. Complete all phases (1-6)")
        print("   3. Then test the dashboard")
        # Fail the test explicitly instead of returning a value
        assert False, "Required data files not found"

    # Test loading articles
    print("\n📊 Testing articles loading...")
    try:
        articles_df = pd.read_csv(articles_path)
        print(f"   ✅ Loaded {len(articles_df)} articles")
        print(f"   ✅ Columns: {list(articles_df.columns)}")

        # Check required columns
        required_columns = ["Title", "Section", "Topic", "Author", "Content"]
        missing_columns = [
            col for col in required_columns if col not in articles_df.columns
        ]

        if missing_columns:
            print(f"   ❌ Missing columns: {missing_columns}")
            assert False, f"Missing required columns: {missing_columns}"
        else:
            print("   ✅ All required columns present")

    except Exception as e:
        print(f"   ❌ Error loading articles: {e}")
        assert False, f"Error loading articles: {e}"

    # Test loading metadata
    print("\n📋 Testing metadata loading...")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("   ✅ Loaded metadata")
        print(f"   ✅ Topics discovered: {metadata['parameters']['best_k']}")
        print(
            f"   ✅ Silhouette score: {metadata['parameters']['silhouette_score']:.3f}"
        )

        # Check required metadata fields
        required_fields = ["parameters", "topics"]
        missing_fields = [field for field in required_fields if field not in metadata]

        if missing_fields:
            print(f"   ❌ Missing metadata fields: {missing_fields}")
            assert False, f"Missing metadata fields: {missing_fields}"
        else:
            print("   ✅ All required metadata fields present")

    except Exception as e:
        print(f"   ❌ Error loading metadata: {e}")
        assert False, f"Error loading metadata: {e}"

    # Test data consistency
    print("\n🔍 Testing data consistency...")
    try:
        # Check if topic numbers match
        max_topic_in_articles = articles_df["Topic"].max()
        topics_in_metadata = len(metadata["topics"])

        print(f"   Max topic in articles: {max_topic_in_articles}")
        print(f"   Topics in metadata: {topics_in_metadata}")

        if max_topic_in_articles >= topics_in_metadata:
            print("   ✅ Topic numbers are consistent")
        else:
            print("   ⚠️  Topic numbers may be inconsistent")

        # Check article count per topic
        topic_counts = articles_df["Topic"].value_counts().sort_index()
        print(f"   Articles per topic: {dict(topic_counts)}")

    except Exception as e:
        print(f"   ❌ Error checking consistency: {e}")
        assert False, f"Error checking consistency: {e}"

    print("\n🎉 All tests passed! Dashboard should work correctly.")


def test_dashboard_imports():
    """Test if dashboard can import all required modules."""
    print("\n📦 Testing dashboard imports...")

    required_modules = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "wordcloud",
        "json",
        "os",
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n❌ Missing modules: {missing_modules}")
        print(f"💡 Install with: pip install {' '.join(missing_modules)}")
        assert False, f"Missing modules: {missing_modules}"

    print("   ✅ All required modules available")


def main():
    """Main test function."""
    print("🚀 Dashboard Test Suite")
    print("=" * 50)

    # Test imports
    imports_ok = test_dashboard_imports()

    # Test data loading
    data_ok = test_data_loading()

    # Summary
    print("\n📋 Test Summary:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   Data Loading: {'✅' if data_ok else '❌'}")

    if imports_ok and data_ok:
        print("\n🎉 Dashboard is ready to run!")
        print("💡 Run with: python run_dashboard.py")
    else:
        print("\n❌ Dashboard needs fixes before running")
        return False

    return True


if __name__ == "__main__":
    main()
