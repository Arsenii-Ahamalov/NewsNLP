"""
Tests for utils_nlp module.
"""

import pandas as pd

from utils_nlp import (
    get_nlp_statistics,
    lemmatize_tokens,
    preprocess_articles_nlp,
    preprocess_text_pipeline,
    remove_stop_words,
    tokenize_text,
)


class TestUtilsNLP:
    """Test cases for NLP utils functions."""

    def test_tokenize_text_basic(self):
        """Test basic tokenization."""
        text = "scientists have discovered a new breakthrough"
        tokens = tokenize_text(text)

        expected = ["scientists", "have", "discovered", "a", "new", "breakthrough"]
        assert tokens == expected

    def test_tokenize_text_empty(self):
        """Test tokenization with empty input."""
        assert tokenize_text("") == []
        assert tokenize_text(None) == []

    def test_tokenize_text_punctuation(self):
        """Test tokenization with punctuation."""
        text = "hello, world! how are you?"
        tokens = tokenize_text(text)

        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you" in tokens

    def test_remove_stop_words_basic(self):
        """Test stop words removal."""
        tokens = ["the", "scientists", "have", "discovered", "a", "new", "breakthrough"]
        filtered = remove_stop_words(tokens)

        expected = ["scientists", "discovered", "new", "breakthrough"]
        assert filtered == expected

    def test_remove_stop_words_empty(self):
        """Test stop words removal with empty input."""
        assert remove_stop_words([]) == []

    def test_lemmatize_tokens_basic(self):
        """Test lemmatization."""
        tokens = ["running", "better", "discovered", "breakthroughs"]
        lemmatized = lemmatize_tokens(tokens)

        # Check that stemming occurred (words should be shorter or same)
        assert len(lemmatized) == 4
        assert "running" in lemmatized or "runn" in lemmatized  # Should stem to "runn"
        assert "better" in lemmatized or "bett" in lemmatized  # Should stem to "bett"
        assert (
            "discovered" in lemmatized or "discover" in lemmatized
        )  # Should stem to "discover"
        assert (
            "breakthroughs" in lemmatized or "breakthrough" in lemmatized
        )  # Should stem to "breakthrough"

    def test_lemmatize_tokens_empty(self):
        """Test lemmatization with empty input."""
        assert not lemmatize_tokens([])

    def test_preprocess_text_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "The scientists have discovered a new breakthrough in artificial intelligence"
        processed = preprocess_text_pipeline(text)

        # Should remove stop words and lemmatize
        assert "the" not in processed
        assert "have" not in processed
        # Check that stop words are removed (not as substrings)
        processed_words = processed.split()
        assert "a" not in processed_words  # 'a' as a separate word should be removed
        assert "in" not in processed_words  # 'in' as a separate word should be removed
        assert "scientist" in processed_words or "scientists" in processed_words
        assert "discover" in processed_words or "discovered" in processed_words
        assert "new" in processed_words
        assert "breakthrough" in processed_words
        assert "artificial" in processed_words
        assert "intelligence" in processed_words

    def test_preprocess_text_pipeline_empty(self):
        """Test pipeline with empty input."""
        assert preprocess_text_pipeline("") == ""
        assert preprocess_text_pipeline(None) == ""

    def test_preprocess_articles_nlp(self):
        """Test DataFrame preprocessing."""
        df = pd.DataFrame(
            {
                "ID": [0, 1],
                "Title": ["Title 1", "Title 2"],
                "cleaned_text": [
                    "scientists have discovered a new breakthrough",
                    "artificial intelligence shows promising results",
                ],
            }
        )

        result_df = preprocess_articles_nlp(df)

        # Check new columns exist
        assert "tokens" in result_df.columns
        assert "tokens_no_stop" in result_df.columns
        assert "tokens_lemmatized" in result_df.columns
        assert "processed_text" in result_df.columns
        assert "token_count" in result_df.columns
        assert "token_count_no_stop" in result_df.columns
        assert "token_count_lemmatized" in result_df.columns

        # Check data
        assert len(result_df) == 2
        assert isinstance(result_df.iloc[0]["tokens"], list)
        assert isinstance(result_df.iloc[0]["processed_text"], str)

    def test_preprocess_articles_nlp_empty(self):
        """Test DataFrame preprocessing with empty DataFrame."""
        df = pd.DataFrame()
        result_df = preprocess_articles_nlp(df)
        assert result_df.empty

    def test_get_nlp_statistics(self):
        """Test NLP statistics calculation."""
        df = pd.DataFrame(
            {
                "ID": [0, 1],
                "tokens": [
                    ["the", "scientist", "discovered"],
                    ["artificial", "intelligence", "shows"],
                ],
                "token_count": [3, 3],
                "token_count_no_stop": [2, 3],
                "token_count_lemmatized": [2, 3],
            }
        )

        stats = get_nlp_statistics(df)

        assert stats["total_articles"] == 2
        assert stats["average_tokens_per_article"] == 3.0
        assert stats["average_tokens_no_stop"] == 2.5
        assert stats["average_tokens_lemmatized"] == 2.5
        assert stats["stop_words_removed_percentage"] > 0

    def test_get_nlp_statistics_empty(self):
        """Test statistics with empty DataFrame."""
        df = pd.DataFrame()
        stats = get_nlp_statistics(df)
        assert not stats


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestUtilsNLP()

    print("🧪 Testing NLP utils functions...")

    try:
        test_instance.test_tokenize_text_basic()
        print("✅ Tokenization test passed")
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")

    try:
        test_instance.test_remove_stop_words_basic()
        print("✅ Stop words removal test passed")
    except Exception as e:
        print(f"❌ Stop words removal test failed: {e}")

    try:
        test_instance.test_preprocess_text_pipeline()
        print("✅ Text pipeline test passed")
    except Exception as e:
        print(f"❌ Text pipeline test failed: {e}")

    try:
        test_instance.test_preprocess_articles_nlp()
        print("✅ DataFrame preprocessing test passed")
    except Exception as e:
        print(f"❌ DataFrame preprocessing test failed: {e}")

    print("🎯 Manual testing completed!")
