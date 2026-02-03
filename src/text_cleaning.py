"""
Text cleaning and preprocessing functions.

This module provides functions to clean and preprocess text data,
including HTML removal, special character handling, and normalization.
"""

import re

import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean individual text by removing HTML, special characters, and normalizing.

    Args:
        text (str): Raw text content from article

    Returns:
        str: Cleaned text ready for further processing
    """
    if text is None or (isinstance(text, str) and not text.strip()):
        return ""

    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()

    return text


def preprocess_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process entire DataFrame by cleaning all text content.

    Args:
        df (pd.DataFrame): DataFrame with 'Content' column from Phase 2

    Returns:
        pd.DataFrame: DataFrame with additional 'cleaned_text' column
    """
    if df.empty:
        return df

    processed_df = df.copy()
    processed_df["cleaned_text"] = processed_df["Content"].apply(clean_text)
    processed_df["cleaned_text"] = processed_df["cleaned_text"].fillna("")
    processed_df["text_length"] = processed_df["cleaned_text"].str.len()
    processed_df["word_count"] = processed_df["cleaned_text"].str.split().str.len()

    return processed_df


def get_text_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about the cleaned text data.

    Args:
        df (pd.DataFrame): DataFrame with 'cleaned_text' column

    Returns:
        dict: Statistics about the text data
    """
    if df.empty or "cleaned_text" not in df.columns:
        return {}

    stats = {
        "total_articles": len(df),
        "articles_with_content": len(df[df["cleaned_text"].str.len() > 0]),
        "average_text_length": (
            df["text_length"].mean() if "text_length" in df.columns else 0
        ),
        "average_word_count": (
            df["word_count"].mean() if "word_count" in df.columns else 0
        ),
        "min_text_length": (
            df["text_length"].min() if "text_length" in df.columns else 0
        ),
        "max_text_length": (
            df["text_length"].max() if "text_length" in df.columns else 0
        ),
    }

    return stats
