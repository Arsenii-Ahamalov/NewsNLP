"""
Article Database Management

Functions to manage the article database, including merging new articles
with existing ones and maintaining data consistency.
"""

import json
import os
import time
from datetime import datetime

import pandas as pd


def merge_articles_with_database(
    new_articles: pd.DataFrame,
    database_path: str = "data/processed/articles_with_topics.csv",
    backup: bool = True,
) -> pd.DataFrame:
    """
    Merge new articles with existing database.

    Args:
        new_articles: DataFrame with new articles to add
        database_path: Path to existing database
        backup: Whether to create backup before merging

    Returns:
        DataFrame with merged articles
    """
    # Check if database exists
    if os.path.exists(database_path):
        # Load existing database
        existing_articles = pd.read_csv(database_path)

        # Create backup if requested
        if backup:
            backup_path = (
                f"{database_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            existing_articles.to_csv(backup_path, index=False)
            print(f"✅ Backup created: {backup_path}")

        # Always append new articles (even if URLs repeat) and ensure unique IDs
        merged_articles = existing_articles.copy()
        new_to_add = new_articles.copy()

        # Reset / assign IDs for new articles so they don't clash
        max_id = (
            merged_articles["ID"].max()
            if "ID" in merged_articles.columns
            else 0
        )
        new_to_add["ID"] = range(max_id + 1, max_id + 1 + len(new_to_add))

        merged_articles = pd.concat(
            [merged_articles, new_to_add], ignore_index=True
        )
        print(f"✅ Added {len(new_to_add)} articles to database")
    else:
        # No existing database, use new articles (and ensure IDs)
        merged_articles = new_articles.copy()
        if "ID" not in merged_articles.columns:
            merged_articles["ID"] = range(1, 1 + len(merged_articles))
        print(f"✅ Created new database with {len(merged_articles)} articles")

    return merged_articles


def update_article_database(
    new_articles: pd.DataFrame,
    database_path: str = "data/processed/articles_with_topics.csv",
    reprocess_topics: bool = True,
) -> bool:
    """
    Update the article database with new articles and optionally reprocess topics.

    Args:
        new_articles: DataFrame with new articles
        database_path: Path to database
        reprocess_topics: Whether to reprocess topic modeling

    Returns:
        True if successful, False otherwise
    """
    try:
        # Merge articles
        merged_articles = merge_articles_with_database(new_articles, database_path)
        # Ensure all articles have summaries
        if (
            "summary" not in merged_articles.columns
            or merged_articles["summary"].isna().any()
            or (merged_articles["summary"].astype(str).str.strip() == "").any()
        ):
            print("📝 Generating summaries for articles without summaries...")
            # Import summarization function
            try:
                from textrank import summarize_articles
            except ImportError:
                try:
                    from src.textrank import summarize_articles
                except ImportError:
                    print("⚠️ Could not import summarize_articles, skipping summary generation")
                    summarize_articles = None
            if summarize_articles:
                # Find articles without summaries
                mask_no_summary = (
                    merged_articles["summary"].isna()
                    if "summary" in merged_articles.columns
                    else pd.Series(
                        [True] * len(merged_articles),
                        index=merged_articles.index,
                    )
                ) | (
                    merged_articles["summary"].astype(str).str.strip() == ""
                    if "summary" in merged_articles.columns
                    else pd.Series(
                        [True] * len(merged_articles),
                        index=merged_articles.index,
                    )
                )
                if mask_no_summary.any():
                    articles_to_summarize = merged_articles[mask_no_summary].copy()
                    # Ensure required columns exist
                    if (
                        "Content" not in articles_to_summarize.columns
                        and "cleaned_text" in articles_to_summarize.columns
                    ):
                        articles_to_summarize["Content"] = articles_to_summarize[
                            "cleaned_text"
                        ]
                    # Generate summaries
                    summarized = summarize_articles(articles_to_summarize)
                    # Update merged articles with summaries
                    for idx in summarized.index:
                        if idx in merged_articles.index:
                            merged_articles.at[idx, "summary"] = summarized.at[idx, "summary"]
                            merged_articles.at[idx, "summary_length"] = summarized.at[
                                idx, "summary_length"
                            ]
                            if "compression_ratio" in summarized.columns:
                                merged_articles.at[idx, "compression_ratio"] = (
                                    summarized.at[idx, "compression_ratio"]
                                )
                    print(f"✅ Generated summaries for {len(summarized)} articles")
        # Save updated database
        merged_articles.to_csv(database_path, index=False)
        print(f"✅ Database updated: {database_path}")
        if (
            reprocess_topics and len(merged_articles) > 10
        ):  # Need enough articles for clustering
            print("🔄 Reprocessing topic modeling...")
            # Import topic modeling functions
            from sklearn.metrics import silhouette_score
            from topic_modeling import (
                assign_topics_to_articles,
                cluster_articles,
                create_tfidf_matrix,
                get_top_keywords_per_cluster,
            )
            # Extract processed text
            texts = merged_articles["processed_text"].fillna("").astype(str).tolist()
            # Create TF-IDF matrix
            tfidf_matrix, feature_names = create_tfidf_matrix(texts)
            # Test different K values
            k_values = [2, 3, 4]
            silhouette_scores = []
            for k in k_values:
                labels, _ = cluster_articles(
                    tfidf_matrix, clusters_count=k, random_state=42
                )
                if len(set(labels)) > 1:
                    sil_score = silhouette_score(tfidf_matrix, labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
            # Find best K
            best_k_idx = max(
                range(len(silhouette_scores)), key=lambda i: silhouette_scores[i]
            )
            best_k = k_values[best_k_idx]
            best_score = silhouette_scores[best_k_idx]
            # Final clustering
            final_labels, km = cluster_articles(
                tfidf_matrix, clusters_count=best_k, random_state=42
            )
            # Extract keywords
            top_keywords = get_top_keywords_per_cluster(
                km.cluster_centers_, feature_names, top_n=10
            )
            # Assign topics
            final_articles = assign_topics_to_articles(merged_articles, final_labels)
            # Preserve summaries from merged_articles (they might be lost in assign_topics_to_articles)
            if "summary" in merged_articles.columns:
                # Ensure summary columns exist in final_articles
                if "summary" not in final_articles.columns:
                    final_articles["summary"] = ""
                if "summary_length" not in final_articles.columns:
                    final_articles["summary_length"] = 0
                if "compression_ratio" not in final_articles.columns:
                    final_articles["compression_ratio"] = 0.0
                # Copy summaries from merged_articles to final_articles by matching ID
                if "ID" in merged_articles.columns and "ID" in final_articles.columns:
                    # Create mapping by ID
                    for idx_final in final_articles.index:
                        article_id = final_articles.at[idx_final, "ID"]
                        # Find matching row in merged_articles by ID
                        matching_rows = merged_articles[merged_articles["ID"] == article_id]
                        if len(matching_rows) > 0:
                            idx_merged = matching_rows.index[0]
                            summary_val = merged_articles.at[idx_merged, "summary"]
                            if pd.notna(summary_val) and str(summary_val).strip():
                                final_articles.at[idx_final, "summary"] = summary_val
                                final_articles.at[idx_final, "summary_length"] = merged_articles.at[idx_merged, "summary_length"]
                                if "compression_ratio" in merged_articles.columns:
                                    final_articles.at[idx_final, "compression_ratio"] = merged_articles.at[idx_merged, "compression_ratio"]
                else:
                    # Fallback: copy by index if IDs don't match
                    for idx in final_articles.index:
                        if idx in merged_articles.index:
                            summary_val = merged_articles.at[idx, "summary"]
                            if pd.notna(summary_val) and str(summary_val).strip():
                                final_articles.at[idx, "summary"] = summary_val
                                final_articles.at[idx, "summary_length"] = merged_articles.at[idx, "summary_length"]
                                if "compression_ratio" in merged_articles.columns:
                                    final_articles.at[idx, "compression_ratio"] = merged_articles.at[idx, "compression_ratio"]
            # Save updated articles with topics
            final_articles.to_csv(database_path, index=False)
            # Update metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "phase": "Database Update with Topic Reprocessing",
                "parameters": {
                    "best_k": int(best_k),
                    "silhouette_score": float(best_score),
                    "total_articles": len(final_articles),
                    "total_features": len(feature_names),
                    "k_values_tested": k_values,
                    "silhouette_scores": [float(s) for s in silhouette_scores],
                },
                "topics": {
                    f"topic_{i}": {
                        "keywords": keywords[:10],
                        "article_count": int((final_articles["Topic"] == i).sum()),
                    }
                    for i, keywords in enumerate(top_keywords)
                },
            }
            metadata_path = "data/processed/topic_modeling_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(
                f"✅ Topic modeling updated: {best_k} topics, score: {best_score:.3f}"
            )
        return True

    except Exception as e:
        print(f"❌ Error updating database: {e}")
        return False


def regenerate_summaries_for_database(
    database_path: str = "data/processed/articles_with_topics.csv",
) -> bool:
    """
    Regenerate summaries for all articles in the database that don't have them.
    Args:
        database_path: Path to database
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(database_path):
            print(f"❌ Database not found: {database_path}")
            return False
        # Load database
        articles_df = pd.read_csv(database_path)
        if articles_df.empty:
            print("ℹ️ Database is empty")
            return True
        # Import summarization function
        try:
            from textrank import summarize_articles
        except ImportError:
            try:
                from src.textrank import summarize_articles
            except ImportError:
                print("❌ Could not import summarize_articles")
                return False
        # Find articles without summaries
        has_summary_col = "summary" in articles_df.columns
        if has_summary_col:
            mask_no_summary = (
                articles_df["summary"].isna()
                | (articles_df["summary"].astype(str).str.strip() == "")
            )
        else:
            mask_no_summary = pd.Series(
                [True] * len(articles_df), index=articles_df.index
            )
        if not mask_no_summary.any():
            print("✅ All articles already have summaries")
            return True
        articles_to_summarize = articles_df[mask_no_summary].copy()
        print(f"📝 Generating summaries for {len(articles_to_summarize)} articles...")
        # Ensure Content column exists
        if "Content" not in articles_to_summarize.columns:
            if "cleaned_text" in articles_to_summarize.columns:
                articles_to_summarize["Content"] = articles_to_summarize["cleaned_text"]
            else:
                print("❌ No Content or cleaned_text column found")
                return False
        # Generate summaries
        print(
            "📊 Columns in articles_to_summarize: "
            f"{list(articles_to_summarize.columns)}"
        )
        print(
            "📊 Sample Content length: "
            f"{articles_to_summarize['Content'].iloc[0] if 'Content' in articles_to_summarize.columns and len(articles_to_summarize) > 0 else 'N/A'}"
        )
        summarized = summarize_articles(articles_to_summarize)
        print(
            "📊 Generated summaries: "
            f"{summarized['summary'].notna().sum()} out of {len(summarized)}"
        )
        print(
            "📊 Sample summary: "
            f"{summarized['summary'].iloc[0][:100] if len(summarized) > 0 and len(str(summarized['summary'].iloc[0])) > 0 else 'EMPTY'}"
        )
        # Update original dataframe - use ID column for matching if available
        if "ID" in articles_to_summarize.columns and "ID" in articles_df.columns:
            # Match by ID instead of index
            id_to_summary = dict(zip(summarized["ID"], summarized["summary"]))
            id_to_length = dict(zip(summarized["ID"], summarized["summary_length"]))
            id_to_ratio = (
                dict(zip(summarized["ID"], summarized["compression_ratio"]))
                if "compression_ratio" in summarized.columns
                else {}
            )
            if "summary" not in articles_df.columns:
                articles_df["summary"] = ""
            if "summary_length" not in articles_df.columns:
                articles_df["summary_length"] = 0
            if "compression_ratio" not in articles_df.columns:
                articles_df["compression_ratio"] = 0.0
            updated_count = 0
            for idx_row in articles_df.index:
                article_id = articles_df.at[idx_row, "ID"]
                if article_id in id_to_summary:
                    articles_df.at[idx_row, "summary"] = id_to_summary[article_id]
                    articles_df.at[idx_row, "summary_length"] = id_to_length[article_id]
                    if article_id in id_to_ratio:
                        articles_df.at[idx_row, "compression_ratio"] = id_to_ratio[
                            article_id
                        ]
                    updated_count += 1
            print(f"📊 Updated {updated_count} articles by ID")
        else:
            # Fallback: match by index
            for idx in summarized.index:
                if idx in articles_df.index:
                    if "summary" not in articles_df.columns:
                        articles_df["summary"] = ""
                    if "summary_length" not in articles_df.columns:
                        articles_df["summary_length"] = 0
                    if "compression_ratio" not in articles_df.columns:
                        articles_df["compression_ratio"] = 0.0
                    articles_df.at[idx, "summary"] = summarized.at[idx, "summary"]
                    articles_df.at[idx, "summary_length"] = summarized.at[
                        idx, "summary_length"
                    ]
                    if "compression_ratio" in summarized.columns:
                        articles_df.at[idx, "compression_ratio"] = summarized.at[
                            idx, "compression_ratio"
                        ]
        # Save updated database
        articles_df.to_csv(database_path, index=False)
        print(f"✅ Regenerated summaries for {len(summarized)} articles")
        return True
    except Exception as e:
        print(f"❌ Error regenerating summaries: {e}")
        return False


def get_database_stats(
    database_path: str = "data/processed/articles_with_topics.csv",
) -> dict:
    """
    Get statistics about the article database.

    Args:
        database_path: Path to database

    Returns:
        Dictionary with database statistics
    """
    if not os.path.exists(database_path):
        return {"error": "Database not found"}
    try:
        # Force reload from disk to get latest data
        time.sleep(0.1)  # Small delay to ensure file is written
        df = pd.read_csv(database_path)

        stats = {
            "total_articles": len(df),
            "sections": (
                df["Section"].value_counts().to_dict()
                if "Section" in df.columns
                else {}
            ),
            "topics": (
                df["Topic"].value_counts().to_dict() if "Topic" in df.columns else {}
            ),
            "date_range": {
                "earliest": (
                    df["Published Date"].min()
                    if "Published Date" in df.columns
                    else None
                ),
                "latest": (
                    df["Published Date"].max()
                    if "Published Date" in df.columns
                    else None
                ),
            },
            "has_summaries": "summary" in df.columns,
            "has_topics": "Topic" in df.columns,
            "last_updated": datetime.now().isoformat(),  # Current timestamp
        }
        return stats
    except Exception as e:
        return {"error": f"Error reading database: {e}"}
