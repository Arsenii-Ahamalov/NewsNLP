#!/usr/bin/env python3
"""
Improved Topic Modeling Script

This script regenerates topic modeling with improved parameters:
- Better stop words filtering
- Fewer clusters to avoid over-clustering
- Improved keyword extraction
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.topic_modeling import (
    assign_topics_to_articles,
    cluster_articles,
    create_tfidf_matrix,
    get_top_keywords_per_cluster,
)


def main():
    """Main function to regenerate topic modeling."""
    print("🔄 Regenerating Topic Modeling with Improved Parameters")
    print("=" * 60)
    # Load summarized data
    try:
        df_summarized = pd.read_csv('data/processed/articles_summarized.csv')
        print(f"✅ Loaded {len(df_summarized)} articles")
    except FileNotFoundError:
        print("❌ Summarized data not found. Please run the data processing pipeline first.")
        return False
    # Extract processed text for topic modeling
    texts = df_summarized["processed_text"].fillna("").astype(str).tolist()
    print(f"📝 Processing {len(texts)} texts for topic modeling")
    # Create TF-IDF matrix with improved parameters
    print("🔢 Creating TF-IDF matrix...")
    tfidf_matrix, feature_names = create_tfidf_matrix(texts)
    print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"✅ Features: {len(feature_names)}")
    # Use K close to the earlier configuration (7 topics)
    print("\n🎯 Testing K values for optimal clustering...")
    # We keep the silhouette-based selection logic but only test K=7,
    # so the final number of topics matches the previous 7-topic setup.
    k_values = [7]
    silhouette_scores = []

    for k in k_values:
        print(f"\nTesting K={k}...")

        # Cluster articles
        labels, _ = cluster_articles(tfidf_matrix, clusters_count=k, random_state=42)

        # Calculate silhouette score
        if len(set(labels)) > 1:
            sil_score = silhouette_score(tfidf_matrix, labels)
            silhouette_scores.append(sil_score)
            print(f"Silhouette score: {sil_score:.3f}")
        else:
            print("Cannot calculate silhouette score (only 1 cluster)")
            silhouette_scores.append(0)

    # Find best K
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_k_idx]
    best_score = silhouette_scores[best_k_idx]

    print("\n🎉 BEST K SELECTION:")
    print(f"Best K: {best_k}")
    print(f"Best silhouette score: {best_score:.3f}")
    print(f"All scores: {dict(zip(k_values, silhouette_scores))}")

    # Perform final clustering with best K
    print(f"\n🔧 Final clustering with K={best_k}...")
    final_labels, km = cluster_articles(tfidf_matrix, clusters_count=best_k, random_state=42)
    # Extract top keywords
    print("🔤 Extracting top keywords...")
    top_keywords = get_top_keywords_per_cluster(km.cluster_centers_, feature_names, top_n=10)
    # Assign topics to articles
    print("📰 Assigning topics to articles...")
    df_with_topics = assign_topics_to_articles(df_summarized, final_labels)
    # Display results
    print("\n📊 TOPIC ANALYSIS RESULTS:")
    print(f"Total articles: {len(df_with_topics)}")
    print(f"Topics discovered: {best_k}")
    print(f"Silhouette score: {best_score:.3f}")
    print("\n📈 Articles per topic:")
    topic_counts = df_with_topics["Topic"].value_counts().sort_index()
    for topic, count in topic_counts.items():
        print(f"  Topic {topic}: {count} articles")
    print("\n🔤 Top keywords per topic:")
    for i, keywords in enumerate(top_keywords):
        print(f"  Topic {i}: {', '.join(keywords[:5])}")  # Show top 5
    # Save results
    print("\n💾 Saving results...")

    # Save articles with topics
    final_path = "data/processed/articles_with_topics.csv"
    df_with_topics.to_csv(final_path, index=False)
    print(f"✅ Articles with topics saved to: {final_path}")

    # Create and save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 5 - Improved Topic Modeling",
        "parameters": {
            "best_k": int(best_k),
            "silhouette_score": float(best_score),
            "total_articles": len(df_with_topics),
            "total_features": len(feature_names),
            "k_values_tested": k_values,
            "silhouette_scores": [float(s) for s in silhouette_scores],
        },
        "topics": {
            f"topic_{i}": {
                "keywords": keywords[:10],  # Top 10 keywords
                "article_count": int((df_with_topics["Topic"] == i).sum()),
            }
            for i, keywords in enumerate(top_keywords)
        },
    }

    metadata_path = "data/processed/topic_modeling_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"✅ Metadata saved to: {metadata_path}")

    print("\n🎉 IMPROVED TOPIC MODELING COMPLETE!")
    print("📁 Results saved in data/processed/")
    print("🚀 Ready to run the dashboard!")

    return True


if __name__ == "__main__":
    main()
