#!/usr/bin/env python3
"""
Regenerate all plots with topic names instead of numbers.
"""

import json
import os
import sys

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plots import get_topic_names_from_sections, save_all_plots  # noqa: E402
from src.topic_modeling import create_tfidf_matrix  # noqa: E402


def main():
    """Regenerate all plots with topic names."""
    print("🔄 Regenerating plots with topic names...")

    # Load data
    print("📂 Loading data...")
    df = pd.read_csv("data/processed/articles_with_topics.csv")

    # Load metadata
    with open("data/processed/topic_modeling_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"✅ Loaded {len(df)} articles with {metadata['parameters']['best_k']} topics")
    # Get topic names from sections
    print("🏷️ Generating topic names from sections...")
    topic_to_name = get_topic_names_from_sections(df, metadata)
    print(f"✅ Topic names: {topic_to_name}")

    # Extract processed text
    texts = df["processed_text"].fillna("").astype(str).tolist()

    # Create TF-IDF matrix
    print("🔤 Creating TF-IDF matrix...")
    tfidf_matrix, feature_names = create_tfidf_matrix(texts)

    # Get labels from dataframe
    labels = df["Topic"].values

    # Get top keywords directly from metadata
    print("🔑 Extracting keywords...")
    top_keywords = []
    for i in range(metadata["parameters"]["best_k"]):
        topic_key = f"topic_{i}"
        if topic_key in metadata["topics"]:
            top_keywords.append(metadata["topics"][topic_key]["keywords"][:10])
        else:
            top_keywords.append([])

    # Regenerate plots with topic names
    print("📊 Generating plots...")
    saved_files = save_all_plots(
        df=df,
        top_keywords=top_keywords,
        tfidf_matrix=tfidf_matrix,
        labels=labels,
        feature_names=feature_names,
        topic_to_name=topic_to_name,
    )

    print("\n✅ All plots regenerated successfully!")
    print("📁 Saved files:")
    for name, path in saved_files.items():
        print(f"   - {name}: {path}")

    print("\n💡 You can now regenerate the PDF report with updated plots.")


if __name__ == "__main__":
    main()
