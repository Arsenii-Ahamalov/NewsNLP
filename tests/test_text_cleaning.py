"""
Tests for text_cleaning module.
"""

import pandas as pd

from text_cleaning import clean_text, get_text_statistics, preprocess_articles


class TestTextCleaning:
    """Test cases for text cleaning functions."""

    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        test_text = (
            "Scientists have discovered a new breakthrough in AI!!! "
            "The research shows promising results... Dr. Smith said: "
            "'This is significant!'"
        )
        expected = (
            "scientists have discovered a new breakthrough in ai the research "
            "shows promising results dr smith said this is significant"
        )

        result = clean_text(test_text)
        assert result == expected

    def test_clean_text_html_removal(self):
        """Test HTML tag removal."""
        test_text = "<div>This is <b>bold</b> text with <p>paragraphs</p></div>"
        expected = "this is bold text with paragraphs"

        result = clean_text(test_text)
        assert result == expected

    def test_clean_text_special_characters(self):
        """Test special character removal."""
        test_text = "Hello @world! #hashtag $money %percent &symbol"
        expected = "hello world hashtag money percent symbol"

        result = clean_text(test_text)
        assert result == expected

    def test_clean_text_punctuation_normalization(self):
        """Test punctuation normalization."""
        test_text = "Multiple... dots,,, commas!!! exclamations??? questions"
        expected = "multiple dots commas exclamations questions"

        result = clean_text(test_text)
        assert result == expected

    def test_clean_text_whitespace_normalization(self):
        """Test whitespace normalization."""
        test_text = "Multiple    spaces\tand\nnewlines"
        expected = "multiple spaces and newlines"

        result = clean_text(test_text)
        assert result == expected

    def test_clean_text_empty_input(self):
        """Test handling of empty and None inputs."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text(pd.NA) == ""

    def test_clean_text_case_conversion(self):
        """Test case conversion to lowercase."""
        test_text = "UPPERCASE and lowercase MiXeD cAsE"
        expected = "uppercase and lowercase mixed case"

        result = clean_text(test_text)
        assert result == expected

    def test_preprocess_articles_basic(self):
        """Test basic DataFrame preprocessing."""
        df = pd.DataFrame(
            {
                "ID": [0, 1, 2],
                "Title": ["Title 1", "Title 2", "Title 3"],
                "Content": [
                    "This is article 1 content.",
                    "This is article 2 content!!!",
                    "This is article 3 content...",
                ],
            }
        )

        result_df = preprocess_articles(df)

        assert "cleaned_text" in result_df.columns
        assert "text_length" in result_df.columns
        assert "word_count" in result_df.columns
        assert len(result_df) == 3

        assert result_df.iloc[0]["cleaned_text"] == "this is article 1 content"
        assert result_df.iloc[1]["cleaned_text"] == "this is article 2 content"
        assert result_df.iloc[2]["cleaned_text"] == "this is article 3 content"

    def test_preprocess_articles_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result_df = preprocess_articles(df)

        assert result_df.empty

    def test_preprocess_articles_missing_content(self):
        """Test handling of missing content."""
        df = pd.DataFrame(
            {
                "ID": [0, 1],
                "Title": ["Title 1", "Title 2"],
                "Content": ["Valid content", None],
            }
        )

        result_df = preprocess_articles(df)

        assert result_df.iloc[0]["cleaned_text"] == "valid content"
        assert result_df.iloc[1]["cleaned_text"] == ""

    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        df = pd.DataFrame(
            {
                "ID": [0, 1, 2],
                "Content": [
                    "This is article 1 content.",
                    "This is article 2 content with more words.",
                    "Short.",
                ],
            }
        )

        processed_df = preprocess_articles(df)
        stats = get_text_statistics(processed_df)

        assert stats["total_articles"] == 3
        assert stats["articles_with_content"] == 3
        assert stats["average_text_length"] > 0
        assert stats["average_word_count"] > 0
        assert stats["min_text_length"] > 0
        assert stats["max_text_length"] > 0

    def test_get_text_statistics_empty_dataframe(self):
        """Test statistics with empty DataFrame."""
        df = pd.DataFrame()
        stats = get_text_statistics(df)

        assert not stats

    def test_get_text_statistics_no_cleaned_text_column(self):
        """Test statistics without cleaned_text column."""
        df = pd.DataFrame({"ID": [0, 1], "Content": ["Content 1", "Content 2"]})

        stats = get_text_statistics(df)
        assert not stats


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestTextCleaning()

    print("🧪 Testing text cleaning functions...")

    try:
        test_instance.test_clean_text_basic()
        print("✅ Basic text cleaning test passed")
    except Exception as e:
        print(f"❌ Basic text cleaning test failed: {e}")

    try:
        test_instance.test_clean_text_html_removal()
        print("✅ HTML removal test passed")
    except Exception as e:
        print(f"❌ HTML removal test failed: {e}")

    try:
        test_instance.test_preprocess_articles_basic()
        print("✅ DataFrame preprocessing test passed")
    except Exception as e:
        print(f"❌ DataFrame preprocessing test failed: {e}")

    try:
        test_instance.test_get_text_statistics()
        print("✅ Text statistics test passed")
    except Exception as e:
        print(f"❌ Text statistics test failed: {e}")

    print("🎯 Manual testing completed!")
