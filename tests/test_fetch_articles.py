"""
Tests for fetch_articles module.
"""

from fetch_articles import fetch_articles_from_api


class TestFetchArticles:
    """Test cases for fetch_articles_from_api function."""

    def test_basic_fetch(self):
        """Test basic article fetching without filters."""
        result = fetch_articles_from_api(number_of_aricles=3)

        assert result is not None
        assert "response" in result
        assert result["response"]["status"] == "ok"
        assert "results" in result["response"]
        assert len(result["response"]["results"]) <= 3

    def test_query_search(self):
        """Test fetching articles with search query."""
        result = fetch_articles_from_api(query="technology", number_of_aricles=2)

        assert result is not None
        assert result["response"]["status"] == "ok"
        assert len(result["response"]["results"]) <= 2

    def test_section_filter(self):
        """Test fetching articles from specific section."""
        result = fetch_articles_from_api(section="technology", number_of_aricles=2)

        assert result is not None
        assert result["response"]["status"] == "ok"
        assert len(result["response"]["results"]) <= 2

    def test_date_range(self):
        """Test fetching articles with date range."""
        result = fetch_articles_from_api(
            section="business",
            from_date="2024-01-01",
            to_date="2024-01-15",
            number_of_aricles=2,
        )

        assert result is not None
        assert result["response"]["status"] == "ok"
        assert len(result["response"]["results"]) <= 2

    def test_combined_filters(self):
        """Test fetching articles with multiple filters."""
        result = fetch_articles_from_api(
            query="artificial intelligence", section="technology", number_of_aricles=1
        )

        assert result is not None
        assert result["response"]["status"] == "ok"
        assert len(result["response"]["results"]) <= 1

    def test_empty_parameters(self):
        """Test that None parameters don't cause errors."""
        result = fetch_articles_from_api(
            query=None, section=None, from_date=None, to_date=None, number_of_aricles=1
        )

        assert result is not None
        assert result["response"]["status"] == "ok"

    def test_response_structure(self):
        """Test that response has expected structure."""
        result = fetch_articles_from_api(number_of_aricles=1)

        assert "response" in result
        response = result["response"]
        assert "status" in response
        assert "total" in response
        assert "results" in response

        if response["results"]:
            article = response["results"][0]
            expected_fields = ["id", "webTitle", "webUrl", "webPublicationDate"]
            for field in expected_fields:
                assert field in article, f"Missing field: {field}"


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestFetchArticles()

    print("🧪 Testing fetch_articles_from_api function...")

    try:
        test_instance.test_basic_fetch()
        print("✅ Basic fetch test passed")
    except Exception as e:
        print(f"❌ Basic fetch test failed: {e}")

    try:
        test_instance.test_query_search()
        print("✅ Query search test passed")
    except Exception as e:
        print(f"❌ Query search test failed: {e}")

    try:
        test_instance.test_section_filter()
        print("✅ Section filter test passed")
    except Exception as e:
        print(f"❌ Section filter test failed: {e}")

    try:
        test_instance.test_response_structure()
        print("✅ Response structure test passed")
    except Exception as e:
        print(f"❌ Response structure test failed: {e}")

    print("🎯 Manual testing completed!")
