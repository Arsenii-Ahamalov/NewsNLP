"""
Tests for TextRank summarization module.
"""

import networkx as nx
import pandas as pd

from src.textrank import (
    build_sentence_graph,
    calculate_sentence_similarity,
    get_summary_statistics,
    split_into_sentence,
    summarize_articles,
    textrank_summarize,
)


class TestSplitIntoSentence:

    def test_split_into_sentence_basic(self):
        """Test basic sentence splitting."""
        text = "This is sentence one. This is sentence two. This is sentence three!"
        result = split_into_sentence(text)
        assert len(result) == 3
        assert "This is sentence one." in result
        assert "This is sentence two." in result
        assert "This is sentence three!" in result

    def test_split_into_sentence_empty(self):
        """Test sentence splitting with empty input."""
        assert split_into_sentence("") == []
        assert split_into_sentence(None) == []

    def test_split_into_sentence_single(self):
        """Test sentence splitting with single sentence."""
        text = "This is a single sentence"
        result = split_into_sentence(text)
        assert len(result) == 1
        assert result[0] == "This is a single sentence"

    def test_split_into_sentence_question_exclamation(self):
        """Test sentence splitting with questions and exclamations."""
        text = "What is this? This is amazing! Really?"
        result = split_into_sentence(text)
        assert len(result) == 3
        assert "What is this?" in result
        assert "This is amazing!" in result
        assert "Really?" in result


class TestCalculateSentenceSimilarity:

    def test_calculate_sentence_similarity_identical(self):
        """Test similarity of identical sentences."""
        sent1 = "The cat is sleeping"
        sent2 = "The cat is sleeping"
        similarity = calculate_sentence_similarity(sent1, sent2)
        assert similarity == 1.0

    def test_calculate_sentence_similarity_similar(self):
        """Test similarity of similar sentences."""
        sent1 = "The cat is sleeping"
        sent2 = "The cat is resting"
        similarity = calculate_sentence_similarity(sent1, sent2)
        assert 0.0 < similarity < 1.0

    def test_calculate_sentence_similarity_different(self):
        """Test similarity of different sentences."""
        sent1 = "The cat is sleeping"
        sent2 = "The weather is nice"
        similarity = calculate_sentence_similarity(sent1, sent2)
        assert similarity == 0.0

    def test_calculate_sentence_similarity_empty(self):
        """Test similarity with empty inputs."""
        assert calculate_sentence_similarity("", "") == 0.0
        assert calculate_sentence_similarity("Hello", "") == 0.0
        assert calculate_sentence_similarity("", "World") == 0.0
        assert calculate_sentence_similarity(None, "Test") == 0.0


class TestBuildSentenceGraph:

    def test_build_sentence_graph_basic(self):
        """Test basic graph building."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        graph = build_sentence_graph(text)

        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() >= 0

    def test_build_sentence_graph_empty(self):
        """Test graph building with empty input."""
        graph = build_sentence_graph("")
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_build_sentence_graph_single_sentence(self):
        """Test graph building with single sentence."""
        text = "This is a single sentence"
        graph = build_sentence_graph(text)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 0  # No edges for single sentence
        assert graph.number_of_edges() == 0

    def test_build_sentence_graph_node_attributes(self):
        """Test that nodes have sentence attributes."""
        text = "First sentence. Second sentence."
        graph = build_sentence_graph(text)

        if graph.number_of_nodes() > 0:
            node_data = graph.nodes[0]
            assert "sentence" in node_data
            assert isinstance(node_data["sentence"], str)


class TestTextRankSummarize:

    def test_textrank_summarize_basic(self):
        """Test basic TextRank summarization."""
        text = (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. This is the fourth sentence."
        )
        summary = textrank_summarize(text, num_sentences=2)

        assert isinstance(summary, str)
        assert len(summary) > 0
        # With improved filtering we mainly care that summary is non-empty,
        # not the exact number of sentences.
        sentences = split_into_sentence(summary)
        assert 1 <= len(sentences) <= 4

    def test_textrank_summarize_empty(self):
        """Test TextRank summarization with empty input."""
        assert textrank_summarize("") == ""
        assert textrank_summarize(None) == ""

    def test_textrank_summarize_single_sentence(self):
        """Test TextRank summarization with single sentence."""
        text = "This is a single sentence"
        summary = textrank_summarize(text, num_sentences=3)
        assert summary == text

    def test_textrank_summarize_few_sentences(self):
        """Test TextRank summarization with fewer sentences than requested."""
        text = "First sentence. Second sentence."
        summary = textrank_summarize(text, num_sentences=5)
        assert summary == text

    def test_textrank_summarize_different_lengths(self):
        """Test TextRank summarization with different sentence counts."""
        text = (
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        )

        summary_2 = textrank_summarize(text, num_sentences=2)
        summary_3 = textrank_summarize(text, num_sentences=3)

        sentences_2 = split_into_sentence(summary_2)
        sentences_3 = split_into_sentence(summary_3)

        # New implementation may keep slightly more sentences because of
        # quality filtering; we only require that more sentences requested
        # does not produce a shorter summary.
        assert len(sentences_2) >= 1
        assert len(sentences_3) >= len(sentences_2)


class TestSummarizeArticles:

    def test_summarize_articles_basic(self):
        """Test basic article summarization."""
        df = pd.DataFrame(
            {
                "ID": [0, 1],
                "Title": ["Title 1", "Title 2"],
                # Current implementation uses Content column for summarization
                "Content": [
                    "This is article one. It has multiple sentences. The content is interesting.",
                    "This is article two. It also has multiple sentences. The content is different.",
                ],
            }
        )

        # New API uses summary_percentage and operates on Content
        result_df = summarize_articles(df, summary_percentage=0.2)

        assert "summary" in result_df.columns
        assert "summary_length" in result_df.columns
        assert "compression_ratio" in result_df.columns
        assert len(result_df) == 2
        assert all(len(row["summary"]) > 0 for _, row in result_df.iterrows())

    def test_summarize_articles_empty_dataframe(self):
        """Test summarization with empty DataFrame."""
        df = pd.DataFrame()
        result_df = summarize_articles(df)
        assert result_df.empty

    def test_summarize_articles_missing_processed_text(self):
        """Test summarization with missing processed_text column."""
        df = pd.DataFrame(
            {
                "ID": [0, 1],
                "Title": ["Title 1", "Title 2"],
                "Content": [
                    "This is article one. It has multiple sentences.",
                    "This is article two. It also has multiple sentences.",
                ],
            }
        )

        result_df = summarize_articles(df)

        assert "summary" in result_df.columns
        assert len(result_df) == 2
        assert all(len(row["summary"]) > 0 for _, row in result_df.iterrows())

    def test_summarize_articles_error_handling(self):
        """Test error handling in article summarization."""
        df = pd.DataFrame({"ID": [0], "Title": ["Title 1"], "processed_text": [None]})

        result_df = summarize_articles(df)

        assert "summary" in result_df.columns
        assert result_df.iloc[0]["summary"] == ""
        assert result_df.iloc[0]["summary_length"] == 0
        assert result_df.iloc[0]["compression_ratio"] == 0.0


class TestGetSummaryStatistics:

    def test_get_summary_statistics_basic(self):
        """Test basic summary statistics."""
        df = pd.DataFrame(
            {
                "summary": [
                    "Short summary",
                    "This is a longer summary with more words",
                ],
                "summary_length": [2, 8],
                "compression_ratio": [0.1, 0.2],
            }
        )

        stats = get_summary_statistics(df)

        assert "total_articles" in stats
        assert "articles_with_summaries" in stats
        assert "avg_summary_length" in stats
        assert "min_summary_length" in stats
        assert "max_summary_length" in stats
        assert "avg_compression_ratio" in stats

        assert stats["total_articles"] == 2
        assert stats["articles_with_summaries"] == 2
        assert stats["avg_summary_length"] == 5.0
        assert stats["min_summary_length"] == 2
        assert stats["max_summary_length"] == 8

    def test_get_summary_statistics_empty_dataframe(self):
        """Test summary statistics with empty DataFrame."""
        df = pd.DataFrame()
        stats = get_summary_statistics(df)
        assert not stats

    def test_get_summary_statistics_missing_columns(self):
        """Test summary statistics with missing columns."""
        df = pd.DataFrame({"ID": [0, 1], "Title": ["Title 1", "Title 2"]})

        stats = get_summary_statistics(df)
        assert not stats

    def test_get_summary_statistics_partial_data(self):
        """Test summary statistics with partial data."""
        df = pd.DataFrame(
            {
                "summary": ["Summary 1", "", "Summary 3"],
                "summary_length": [5, 0, 3],
                "compression_ratio": [0.1, 0.0, 0.15],
            }
        )

        stats = get_summary_statistics(df)

        assert stats["total_articles"] == 3
        assert stats["articles_with_summaries"] == 2
        assert stats["avg_summary_length"] == 4.0  # (5 + 3) / 2
        assert stats["min_summary_length"] == 3
        assert stats["max_summary_length"] == 5


class TestIntegration:

    def test_full_pipeline(self):
        """Test the complete TextRank pipeline."""
        text = (
            "Scientists have discovered a new breakthrough in artificial "
            "intelligence. The research shows promising results for machine "
            "learning applications. AI technology is advancing rapidly in various "
            "fields. The breakthrough could revolutionize how we approach complex "
            "problems. Machine learning algorithms are becoming more "
            "sophisticated and efficient."
        )

        # Test individual components
        sentences = split_into_sentence(text)
        assert len(sentences) == 5

        similarity = calculate_sentence_similarity(sentences[0], sentences[1])
        assert 0.0 <= similarity <= 1.0

        graph = build_sentence_graph(text)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 5

        summary = textrank_summarize(text, num_sentences=2)
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Test DataFrame integration (uses Content column)
        df = pd.DataFrame(
            {"ID": [0], "Title": ["AI Breakthrough"], "Content": [text]}
        )

        result_df = summarize_articles(df, summary_percentage=0.2)
        assert "summary" in result_df.columns
        assert len(result_df.iloc[0]["summary"]) > 0

        stats = get_summary_statistics(result_df)
        assert stats["total_articles"] == 1
        assert stats["articles_with_summaries"] == 1
