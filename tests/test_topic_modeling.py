"""
Tests for topic_modeling module.
"""

import numpy as np
import pandas as pd
import pytest

from topic_modeling import (
    assign_topics_to_articles,
    cluster_articles,
    create_count_vector,
    create_idf_vector,
    create_tfidf_matrix,
    get_top_keywords_per_cluster,
)


class TestCreateCountVector:
    """Test cases for create_count_vector function."""

    def test_create_count_vector_basic(self):
        """Test basic word counting."""
        words = ["the", "cat", "the", "sat"]
        result = create_count_vector(words)

        assert isinstance(result, list)
        assert len(result) == 3  # 3 unique words
        assert 2 in result  # "the" appears twice
        assert 1 in result  # "cat" and "sat" appear once

    def test_create_count_vector_empty(self):
        """Test with empty word list."""
        words = []
        result = create_count_vector(words)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_create_count_vector_case_insensitive(self):
        """Test that function handles case differences."""
        words = ["The", "cat", "the", "CAT"]
        result = create_count_vector(words)

        assert isinstance(result, list)
        assert len(result) == 2  # "the" and "cat" (case insensitive)
        assert 2 in result  # "the" appears twice
        assert 2 in result  # "cat" appears twice

    def test_create_count_vector_single_word(self):
        """Test with single repeated word."""
        words = ["hello", "hello", "hello"]
        result = create_count_vector(words)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 3


class TestCreateIdfVector:
    """Test cases for create_idf_vector function."""

    def test_create_idf_vector_basic(self):
        """Test basic IDF calculation."""
        texts = ["the cat sat", "the dog ran", "the bird flew"]
        unique_words = {"the", "cat", "dog", "bird", "sat", "ran", "flew"}

        result = create_idf_vector(texts, unique_words)

        assert isinstance(result, list)
        assert len(result) == len(unique_words)
        # "the" appears in all 3 texts, so should have high count
        # Other words appear in 1 text each

    def test_create_idf_vector_empty_texts(self):
        """Test with empty texts list."""
        texts = []
        unique_words = {"word1", "word2"}

        result = create_idf_vector(texts, unique_words)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(count == 0 for count in result)

    def test_create_idf_vector_empty_words(self):
        """Test with empty unique words set."""
        texts = ["some text here"]
        unique_words = set()

        result = create_idf_vector(texts, unique_words)

        assert isinstance(result, list)
        assert len(result) == 0


class TestCreateTfidfMatrix:
    """Test cases for create_tfidf_matrix function."""

    def test_create_tfidf_matrix_basic(self):
        """Test basic TF-IDF matrix creation."""
        # Use more documents so current min_df/max_df settings are valid.
        # We include a token 'common' that appears in exactly 3 documents,
        # which satisfies min_df=3 and max_df=0.7 for 5 documents.
        texts = [
            "alpha common",
            "beta common",
            "gamma common",
            "delta",
            "epsilon",
        ]

        result, feature_names = create_tfidf_matrix(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(texts)  # documents
        assert result.shape[1] > 0  # Some number of unique words
        assert len(feature_names) == result.shape[1]

    def test_create_tfidf_matrix_empty_texts(self):
        """Test with empty texts list."""
        texts = []

        result, feature_names = create_tfidf_matrix(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0  # No documents
        assert result.shape[1] == 0  # No features
        assert feature_names == []

    def test_create_tfidf_matrix_single_text(self):
        """Test with single text."""
        texts = ["the cat sat on the mat"]
        # With current vectorizer settings, a single document is not sufficient
        # for a stable TF-IDF vocabulary, so we expect a ValueError.
        with pytest.raises(ValueError):
            create_tfidf_matrix(texts)

    def test_create_tfidf_matrix_dimensions(self):
        """Test that matrix dimensions are correct."""
        texts = [
            "one common alpha",
            "two common beta",
            "three common gamma",
            "four something",
            "five else",
        ]

        result, feature_names = create_tfidf_matrix(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(texts)  # documents
        # Number of columns should equal number of unique words across all texts
        assert result.shape[1] > 0
        assert len(feature_names) == result.shape[1]
        assert isinstance(feature_names, list)


class TestClusterArticles:
    """Test cases for cluster_articles function."""

    def test_cluster_articles_deterministic(self):
        """Same random_state yields identical labels."""
        # Use a small synthetic TF-IDF matrix to avoid dependence on
        # specific vectorizer parameters.
        tfidf = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels1, _ = cluster_articles(tfidf, clusters_count=2, random_state=42)
        labels2, _ = cluster_articles(tfidf, clusters_count=2, random_state=42)
        assert isinstance(labels1, np.ndarray)
        assert (labels1 == labels2).all()

    def test_cluster_articles_label_shape_and_range(self):
        """Labels length equals documents and values are within [0, k-1]."""
        tfidf = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.1],
                [0.0, 1.0],
                [0.2, 0.9],
            ]
        )
        k = 2
        labels, km = cluster_articles(tfidf, clusters_count=k, random_state=0)
        assert len(labels) == tfidf.shape[0]
        assert set(labels).issubset(set(range(k)))
        # sanity: cluster centers shape (k, n_features)
        assert km.cluster_centers_.shape[0] == k
        assert km.cluster_centers_.shape[1] == tfidf.shape[1]

    def test_cluster_articles_separation_on_toy_data(self):
        """Simple toy corpus should separate into two coherent clusters."""
        tfidf = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels, _ = cluster_articles(tfidf, clusters_count=2, random_state=0)
        print("DEBUG labels:", labels)
        # Expect first two together and last two together (label ids may swap)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]


class TestGetTopKeywordsPerCluster:
    """Test cases for get_top_keywords_per_cluster function."""

    def test_get_top_keywords_per_cluster_basic(self):
        """Test basic keyword extraction."""
        # Create mock cluster centers and feature names
        cluster_centers = np.array(
            [
                [0.1, 0.8, 0.3, 0.2],  # Cluster 0: feature 1 is highest
                [0.7, 0.1, 0.4, 0.1],  # Cluster 1: feature 0 is highest
            ]
        )
        feature_names = ["cat", "dog", "bird", "fish"]

        result = get_top_keywords_per_cluster(cluster_centers, feature_names, top_n=2)

        assert isinstance(result, list)
        assert len(result) == 2  # Two clusters
        assert len(result[0]) == 2  # Top 2 keywords for cluster 0
        assert len(result[1]) == 2  # Top 2 keywords for cluster 1
        assert "dog" in result[0]  # Highest weight in cluster 0
        assert "cat" in result[1]  # Highest weight in cluster 1

    def test_get_top_keywords_per_cluster_empty_inputs(self):
        """Test with empty cluster centers or feature names."""
        # Empty cluster centers
        result1 = get_top_keywords_per_cluster(np.array([]), ["word1", "word2"])
        assert not result1

        # Empty feature names
        result2 = get_top_keywords_per_cluster(np.array([[0.1, 0.2]]), [])
        assert not result2

        # Both empty
        result3 = get_top_keywords_per_cluster(np.array([]), [])
        assert not result3

    def test_get_top_keywords_per_cluster_filtering(self):
        """Test filtering of short words and numbers."""
        cluster_centers = np.array(
            [
                [0.1, 0.8, 0.3, 0.2, 0.5, 0.4],  # Multiple features
            ]
        )
        feature_names = ["a", "cat", "1", "dog", "xyz", "bird"]  # Mix of valid/invalid

        result = get_top_keywords_per_cluster(cluster_centers, feature_names, top_n=10)

        assert isinstance(result, list)
        assert len(result) == 1
        keywords = result[0]

        # Should filter out "a" (too short) and "1" (number)
        assert "a" not in keywords
        assert "1" not in keywords
        # Should keep valid words
        assert (
            "cat" in keywords
            or "dog" in keywords
            or "xyz" in keywords
            or "bird" in keywords
        )

    def test_get_top_keywords_per_cluster_top_n_limit(self):
        """Test that top_n parameter limits results correctly."""
        cluster_centers = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 6 features
            ]
        )
        feature_names = ["word1", "word2", "word3", "word4", "word5", "word6"]

        result = get_top_keywords_per_cluster(cluster_centers, feature_names, top_n=3)

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) <= 3  # Should not exceed top_n

    def test_get_top_keywords_per_cluster_alignment(self):
        """Test that keywords align with cluster center weights."""
        cluster_centers = np.array(
            [
                [0.1, 0.9, 0.2],  # Feature 1 should be top keyword
                [0.8, 0.1, 0.3],  # Feature 0 should be top keyword
            ]
        )
        feature_names = ["low", "high", "medium"]

        result = get_top_keywords_per_cluster(cluster_centers, feature_names, top_n=1)

        assert isinstance(result, list)
        assert len(result) == 2
        # First cluster should have "high" as top keyword
        assert result[0][0] == "high"
        # Second cluster should have "low" as top keyword
        assert result[1][0] == "low"


class TestAssignTopicsToArticles:
    """Test cases for assign_topics_to_articles function."""

    def test_assign_topics_to_articles_basic(self):
        """Test basic topic assignment."""
        df = pd.DataFrame(
            {
                "Title": ["Article 1", "Article 2", "Article 3"],
                "Content": ["content 1", "content 2", "content 3"],
            }
        )
        labels = np.array([0, 1, 0])

        result = assign_topics_to_articles(df, labels)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "Topic" in result.columns
        assert list(result["Topic"]) == [0, 1, 0]
        assert list(result["Title"]) == [
            "Article 1",
            "Article 2",
            "Article 3",
        ]  # Original data preserved

    def test_assign_topics_to_articles_length_mismatch(self):
        """Test error when DataFrame and labels have different lengths."""
        df = pd.DataFrame(
            {"Title": ["Article 1", "Article 2"], "Content": ["content 1", "content 2"]}
        )
        labels = np.array([0, 1, 0])  # 3 labels for 2 articles

        with pytest.raises(
            ValueError, match="DataFrame length.*must match labels length"
        ):
            assign_topics_to_articles(df, labels)

    def test_assign_topics_to_articles_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["Title", "Content"])
        labels = np.array([])

        result = assign_topics_to_articles(df, labels)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "Topic" in result.columns

    def test_assign_topics_to_articles_preserves_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame(
            {"Title": ["Article 1", "Article 2"], "Content": ["content 1", "content 2"]}
        )
        labels = np.array([0, 1])

        result = assign_topics_to_articles(df, labels)

        # Original DataFrame should not have Topic column
        assert "Topic" not in df.columns
        # Result should have Topic column
        assert "Topic" in result.columns
        # Original data should be preserved
        assert list(df["Title"]) == list(result["Title"])


if __name__ == "__main__":
    pytest.main([__file__])
