"""
Tests for the article database management helpers.
"""

import pandas as pd

from database_manager import (
    get_database_stats,
    merge_articles_with_database,
    update_article_database,
)


def _make_sample_articles(n: int, start_id: int = 1) -> pd.DataFrame:
    """Create a small DataFrame with the minimal columns used in database_manager."""
    data = []
    for i in range(n):
        article_id = start_id + i
        data.append(
            {
                "ID": article_id,
                "Title": f"Article {article_id}",
                "Section": "Test",
                "Author": "Tester",
                "Url": f"https://example.com/{article_id}",
                "Content": f"Content of article {article_id}",
                "processed_text": f"processed article {article_id}",
                "summary": f"Summary {article_id}",
                "summary_length": 2,
                "compression_ratio": 0.5,
            }
        )
    return pd.DataFrame(data)


def test_merge_articles_with_database_appends_and_assigns_ids(tmp_path):
    """merge_articles_with_database should append new rows and keep/assign IDs."""
    db_path = tmp_path / "articles_with_topics.csv"

    # Existing database with 2 articles
    existing = _make_sample_articles(2, start_id=1)
    existing.to_csv(db_path, index=False)

    # New articles without IDs (simulating fresh processed batch)
    new_articles = _make_sample_articles(2, start_id=100).drop(columns=["ID"])

    merged = merge_articles_with_database(
        new_articles, database_path=str(db_path), backup=False
    )

    # Should now contain 4 rows
    assert len(merged) == 4

    # IDs of original rows must stay 1,2 and new must be 3,4
    assert set(merged["ID"].tolist()) == {1, 2, 3, 4}


def test_update_article_database_writes_to_csv(tmp_path):
    """update_article_database should persist the merged result to the given path."""
    db_path = tmp_path / "articles_with_topics.csv"

    # Start with a small database
    base_df = _make_sample_articles(1, start_id=1)
    base_df.to_csv(db_path, index=False)

    # New batch with one new article
    new_batch = _make_sample_articles(1, start_id=2).drop(columns=["ID"])

    # Do not reprocess topics here to keep the test lightweight
    ok = update_article_database(
        new_batch, database_path=str(db_path), reprocess_topics=False
    )

    assert ok is True

    # CSV on disk should now contain 2 rows
    reloaded = pd.read_csv(db_path)
    assert len(reloaded) == 2


def test_get_database_stats_reads_basic_info(tmp_path):
    """get_database_stats should return counts based on the CSV on disk."""
    db_path = tmp_path / "articles_with_topics.csv"

    df = _make_sample_articles(3, start_id=1)
    # Add simple Topic + Published Date columns to exercise stats fields
    df["Topic"] = [0, 1, 1]
    df["Published Date"] = ["2025-01-01", "2025-01-02", "2025-01-03"]
    df.to_csv(db_path, index=False)

    stats = get_database_stats(database_path=str(db_path))

    assert stats["total_articles"] == 3
    # Sections and topics should be non-empty dicts
    assert stats["sections"]["Test"] == 3
    assert stats["topics"][1] == 2
