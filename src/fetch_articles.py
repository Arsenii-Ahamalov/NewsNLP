"""
Fetch articles from The Guardian API.

This module provides functions to fetch and process articles from The Guardian API.
"""

import re
from typing import Any, Dict, Optional

import pandas as pd
import requests

from config import GUARDIAN_API_KEY, GUARDIAN_BASE_URL


def fetch_articles_from_api(
    query: Optional[str] = None,
    section: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    number_of_aricles: int = 10,
) -> Dict[str, Any]:
    """
    Fetch articles from The Guardian API.

    Args:
        query (Optional[str]): Search query to filter results.
        section (Optional[str]): Section of the news to filter articles (e.g., 'technology').
        from_date (Optional[str]): The start date for results (YYYY-MM-DD).
        to_date (Optional[str]): The end date for results (YYYY-MM-DD).
        number_of_aricles (int): The number of articles to retrieve (default is 10).

    Returns:
        Dict[str, Any]: The JSON response from the Guardian API as a dictionary.

    Raises:
        Exception: If the API request fails (non-200 response code).
    """
    request_params = {
        "q": query,
        "section": section,
        "from-date": from_date,
        "to-date": to_date,
        "page-size": number_of_aricles,
        "api-key": GUARDIAN_API_KEY,
        "show-fields": "headline,body,byline,thumbnail",
    }
    response = requests.get(GUARDIAN_BASE_URL, params=request_params, timeout=30)
    if response.status_code == 200:
        return response.json()
    msg = f"Failed to fetch articles from API: {response.status_code}"
    raise RuntimeError(msg)


def process_articles(response: Dict[str, Any]) -> pd.DataFrame:
    """
    Process a Guardian API response and convert articles into a pandas DataFrame.

    Args:
        response (Dict[str, Any]): The response dictionary from Guardian API.

    Returns:
        pd.DataFrame: DataFrame containing processed article fields:
            - ID: Simple integer counter (0, 1, 2, ...).
            - Title: Article title.
            - Url: Article URL.
            - Section: Section of publication.
            - Published Date: Clean publication date.
            - Author: Author name(s), if available.
            - Content: Clean body content without HTML tags.

    Raises:
        KeyError: If required fields are missing in the article dict.
    """
    articles_list = []

    articles = response.get("response", {}).get("results", [])

    if not isinstance(articles, list):
        articles = []

    for i, article in enumerate(articles):
        fields = article.get("fields", {})

        content = fields.get("body", "")
        if content:
            content = re.sub(r"<[^>]+>", "", content)
            content = " ".join(content.split())

        published_date = article["webPublicationDate"]
        if published_date:
            published_date = published_date.replace("T", " ").replace("Z", "")

        articles_list.append(
            {
                "ID": i,
                "Title": article["webTitle"],
                "Url": article["webUrl"],
                "Section": article["sectionName"],
                "Published Date": published_date,
                "Author": fields.get("byline") or "Unknown",
                "Content": content,
            }
        )

    return pd.DataFrame(articles_list)


def get_articles(
    query: Optional[str] = None,
    section: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    number_of_articles: int = 10,
) -> pd.DataFrame:
    """
    Main function to fetch and process articles from Guardian API.

    This function combines fetch_articles_from_api() and process_articles() to return
    a clean, processed DataFrame ready for analysis.

    Args:
        query (Optional[str]): Search query to filter results.
        section (Optional[str]): Section of the news to filter articles (e.g., 'technology').
        from_date (Optional[str]): The start date for results (YYYY-MM-DD).
        to_date (Optional[str]): The end date for results (YYYY-MM-DD).
        number_of_articles (int): The number of articles to retrieve (default is 10).

    Returns:
        pd.DataFrame: Clean DataFrame with processed articles containing columns:
            - ID: Unique article ID
            - Title: Article title
            - Url: Article URL
            - Section: Section of publication
            - Published Date: Publication date (ISO 8601)
            - Author: Author name(s), if available
            - Content: Body content, if available

    Raises:
        Exception: If the API request fails or processing fails.
    """
    try:
        raw_response = fetch_articles_from_api(
            query=query,
            section=section,
            from_date=from_date,
            to_date=to_date,
            number_of_aricles=number_of_articles,
        )

        processed_df = process_articles(raw_response)

        return processed_df

    except Exception as e:
        raise Exception(f"Failed to get articles: {str(e)}") from e
