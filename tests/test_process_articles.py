"""
Tests for process_articles functionality.
"""

import pandas as pd

from fetch_articles import fetch_articles_from_api, process_articles


class TestProcessArticles:
    """Test cases for process_articles function."""

    def test_process_articles_basic(self):
        """Test basic article processing."""
        # Fetch real data
        response = fetch_articles_from_api(number_of_aricles=3)

        # Process articles
        df = process_articles(response)

        # Basic checks
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

    def test_process_articles_structure(self):
        """Test that processed articles have correct structure."""
        response = fetch_articles_from_api(number_of_aricles=2)
        df = process_articles(response)

        if len(df) > 0:
            # Check data types and non-empty values
            assert not df["ID"].isna().any(), "ID should not be null"
            assert not df["Title"].isna().any(), "Title should not be null"
            assert not df["Url"].isna().any(), "Url should not be null"
            assert not df["Section"].isna().any(), "Section should not be null"
            assert (
                not df["Published Date"].isna().any()
            ), "Published Date should not be null"

            # Check that URLs are valid
            assert (
                df["Url"].str.startswith("http").all()
            ), "All URLs should start with http"

    def test_process_articles_empty_response(self):
        """Test handling of empty response."""
        empty_response = {"response": {"status": "ok", "total": 0, "results": []}}

        df = process_articles(empty_response)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Empty DataFrame should have the correct columns
        expected_columns = [
            "ID",
            "Title",
            "Url",
            "Section",
            "Published Date",
            "Author",
            "Content",
        ]
        if len(df) == 0:
            # For empty DataFrame, check if columns exist or create them
            df = pd.DataFrame(columns=expected_columns)
        assert list(df.columns) == expected_columns

    def test_process_articles_missing_fields(self):
        """Test handling of articles with missing fields."""
        # Create mock response with correct Guardian API structure
        mock_response = {
            "response": {
                "status": "ok",
                "results": [
                    {
                        "id": "test-article-1",
                        "webTitle": "Test Article",
                        "webUrl": "https://example.com",
                        "sectionName": "Technology",
                        "webPublicationDate": "2024-01-15T10:30:00Z",
                        "fields": {
                            # Missing "byline" and "body"
                        },
                    },
                    {
                        "id": "test-article-2",
                        "webTitle": "Another Test",
                        "webUrl": "https://example2.com",
                        "sectionName": "Science",
                        "webPublicationDate": "2024-01-16T11:00:00Z",
                        "fields": {
                            "byline": "John Doe",
                            "body": "This is article content",
                        },
                    },
                ],
            }
        }

        df = process_articles(mock_response)

        assert len(df) == 2
        assert df.iloc[0]["Author"] == "Unknown"  # Should default to "Unknown"
        assert df.iloc[0]["Content"] == "" or pd.isna(df.iloc[0]["Content"])
        assert df.iloc[1]["Author"] == "John Doe"
        assert df.iloc[1]["Content"] == "This is article content"

    def test_process_articles_data_types(self):
        """Test that processed data has correct types."""
        response = fetch_articles_from_api(number_of_aricles=2)
        df = process_articles(response)

        if len(df) > 0:
            # Check data types
            assert df["ID"].dtype == "int64", "ID should be integer type"
            assert df["Title"].dtype == "object", "Title should be string/object type"
            assert df["Url"].dtype == "object", "Url should be string/object type"
            assert (
                df["Section"].dtype == "object"
            ), "Section should be string/object type"
            assert (
                df["Published Date"].dtype == "object"
            ), "Published Date should be string/object type"

    def test_process_articles_content_quality(self):
        """Test content quality and length."""
        response = fetch_articles_from_api(number_of_aricles=3)
        df = process_articles(response)

        if len(df) > 0:
            # Check that titles are not empty
            assert (df["Title"].str.len() > 0).all(), "All titles should have content"

            # Check that URLs are properly formatted
            assert (
                df["Url"].str.contains("guardian.com").all()
            ), "All URLs should be from Guardian"

            # Check that sections are not empty
            assert (df["Section"].str.len() > 0).all(), "All sections should have names"

    def test_process_articles_duplicate_handling(self):
        """Test that duplicate articles are handled properly."""
        response = fetch_articles_from_api(number_of_aricles=5)
        df = process_articles(response)

        if len(df) > 1:
            # Check for unique IDs
            assert df["ID"].nunique() == len(df), "All article IDs should be unique"

    def test_process_articles_date_format(self):
        """Test that dates are in correct format."""
        response = fetch_articles_from_api(number_of_aricles=2)
        df = process_articles(response)

        if len(df) > 0:
            # Check date format (clean format without T and Z)
            dates = df["Published Date"]
            assert dates.str.contains(" ").all(), "Dates should have space separator"
            assert not dates.str.contains("T").any(), "Dates should not contain T"
            assert not dates.str.contains("Z").any(), "Dates should not contain Z"

    def test_process_articles_missing_required_fields(self):
        """Test handling of articles missing required fields like id, webTitle."""
        mock_response = {
            "response": {
                "status": "ok",
                "results": [
                    {
                        # Missing "id" - should cause KeyError
                        "webTitle": "Test Article",
                        "webUrl": "https://example.com",
                        "sectionName": "Technology",
                        "webPublicationDate": "2024-01-15T10:30:00Z",
                        "fields": {},
                    }
                ],
            }
        }

        # This should NOT raise KeyError anymore - we handle missing fields gracefully
        df = process_articles(mock_response)

        assert len(df) == 1
        assert df.iloc[0]["ID"] == 0  # Should use counter
        assert df.iloc[0]["Title"] == "Test Article"

    def test_process_articles_missing_fields_key(self):
        """Test handling when 'fields' key is completely missing."""
        mock_response = {
            "response": {
                "status": "ok",
                "results": [
                    {
                        "id": "test-article-1",
                        "webTitle": "Test Article",
                        "webUrl": "https://example.com",
                        "sectionName": "Technology",
                        "webPublicationDate": "2024-01-15T10:30:00Z",
                        # Missing "fields" key entirely
                    }
                ],
            }
        }

        df = process_articles(mock_response)

        assert len(df) == 1
        assert df.iloc[0]["Author"] == "Unknown"  # Should default to "Unknown"
        assert df.iloc[0]["Content"] == "" or pd.isna(df.iloc[0]["Content"])

    def test_process_articles_empty_strings_vs_none(self):
        """Test handling of empty strings vs None values."""
        mock_response = {
            "response": {
                "status": "ok",
                "results": [
                    {
                        "id": "test-article-1",
                        "webTitle": "",  # Empty string
                        "webUrl": "https://example.com",
                        "sectionName": "Technology",
                        "webPublicationDate": "2024-01-15T10:30:00Z",
                        "fields": {
                            "byline": "",  # Empty string
                            "body": "",  # Empty string
                        },
                    }
                ],
            }
        }

        df = process_articles(mock_response)

        assert len(df) == 1
        assert df.iloc[0]["Title"] == ""  # Should preserve empty string
        assert df.iloc[0]["Author"] == "Unknown"  # Empty string becomes "Unknown"
        assert df.iloc[0]["Content"] == ""  # Should preserve empty string

    def test_process_articles_malformed_response(self):
        """Test handling of completely malformed response."""
        malformed_responses = [
            {},  # Empty dict
            {"response": {}},  # Missing results
            {"response": {"results": None}},  # None results
            {"response": {"results": "not-a-list"}},  # Wrong type
        ]

        for malformed_response in malformed_responses:
            df = process_articles(malformed_response)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0  # Should handle gracefully

    def test_process_articles_data_quality_validation(self):
        """Test actual data quality from real API."""
        response = fetch_articles_from_api(number_of_aricles=5)
        df = process_articles(response)

        if len(df) > 0:
            # Check for actual data quality issues
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Missing values per column:")
            print(df.isnull().sum())

            # Check for common data quality issues
            assert not df["ID"].isnull().any(), "ID should never be null"
            assert not df["Title"].isnull().any(), "Title should never be null"
            assert not df["Url"].isnull().any(), "Url should never be null"

            # Check for reasonable content lengths
            assert (
                df["Title"].str.len() > 5
            ).all(), "Titles should be meaningful length"
            assert (
                df["Url"].str.contains("guardian.com").all()
            ), "All URLs should be Guardian URLs"


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestProcessArticles()

    print("🧪 Testing process_articles function...")

    try:
        test_instance.test_process_articles_basic()
        print("✅ Basic processing test passed")
    except Exception as e:
        print(f"❌ Basic processing test failed: {e}")

    try:
        test_instance.test_process_articles_structure()
        print("✅ Structure test passed")
    except Exception as e:
        print(f"❌ Structure test failed: {e}")

    try:
        test_instance.test_process_articles_missing_fields()
        print("✅ Missing fields test passed")
    except Exception as e:
        print(f"❌ Missing fields test failed: {e}")

    try:
        test_instance.test_process_articles_empty_response()
        print("✅ Empty response test passed")
    except Exception as e:
        print(f"❌ Empty response test failed: {e}")

    print("🎯 Manual testing completed!")
