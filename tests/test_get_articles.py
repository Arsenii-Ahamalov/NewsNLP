"""
Tests for get_articles function.
"""

import pandas as pd

from fetch_articles import get_articles


class TestGetArticles:
    """Test cases for get_articles function."""

    def test_get_articles_basic(self):
        """Test basic article fetching and processing."""
        df = get_articles(number_of_articles=3)

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 3
        assert len(df) > 0

        # Check columns
        expected_columns = [
            "ID",
            "Title",
            "Url",
            "Section",
            "Published Date",
            "Author",
            "Content",
        ]
        assert list(df.columns) == expected_columns

    def test_get_articles_with_query(self):
        """Test fetching articles with search query."""
        df = get_articles(query="technology", number_of_articles=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 2

        if len(df) > 0:
            # Check that titles contain technology-related content
            titles = df["Title"].str.lower()
            assert any(
                "tech" in title for title in titles
            ), "Should find technology-related articles"

    def test_get_articles_with_section(self):
        """Test fetching articles from specific section."""
        df = get_articles(section="technology", number_of_articles=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 2

        if len(df) > 0:
            # All articles should be from technology section
            assert (
                df["Section"] == "Technology"
            ).all(), "All articles should be from Technology section"

    def test_get_articles_with_date_range(self):
        """Test fetching articles with date range."""
        df = get_articles(
            section="business",
            from_date="2024-01-01",
            to_date="2024-01-15",
            number_of_articles=2,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 2

    def test_get_articles_combined_filters(self):
        """Test fetching articles with multiple filters."""
        df = get_articles(
            query="artificial intelligence", section="technology", number_of_articles=1
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 1

    def test_get_articles_empty_parameters(self):
        """Test that None parameters work correctly."""
        df = get_articles(
            query=None, section=None, from_date=None, to_date=None, number_of_articles=1
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 1

    def test_get_articles_data_quality(self):
        """Test data quality of returned articles."""
        df = get_articles(number_of_articles=3)

        if len(df) > 0:
            # Check for data quality
            assert not df["ID"].isnull().any(), "ID should never be null"
            assert not df["Title"].isnull().any(), "Title should never be null"
            assert not df["Url"].isnull().any(), "Url should never be null"
            assert not df["Section"].isnull().any(), "Section should never be null"
            assert (
                not df["Published Date"].isnull().any()
            ), "Published Date should never be null"

            # Check URL format
            assert (
                df["Url"].str.contains("guardian.com").all()
            ), "All URLs should be Guardian URLs"

            # Check date format (clean format without T and Z)
            dates = df["Published Date"]
            assert dates.str.contains(" ").all(), "Dates should have space separator"
            assert not dates.str.contains("T").any(), "Dates should not contain T"
            assert not dates.str.contains("Z").any(), "Dates should not contain Z"


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestGetArticles()

    print("🧪 Testing get_articles function...")

    try:
        test_instance.test_get_articles_basic()
        print("✅ Basic get_articles test passed")
    except Exception as e:
        print(f"❌ Basic get_articles test failed: {e}")

    try:
        test_instance.test_get_articles_with_query()
        print("✅ Query test passed")
    except Exception as e:
        print(f"❌ Query test failed: {e}")

    try:
        test_instance.test_get_articles_with_section()
        print("✅ Section test passed")
    except Exception as e:
        print(f"❌ Section test failed: {e}")

    try:
        test_instance.test_get_articles_data_quality()
        print("✅ Data quality test passed")
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")

    print("🎯 Manual testing completed!")
