"""
NLP utility functions for text preprocessing.

This module provides functions for tokenization, lemmatization, and
other NLP preprocessing tasks.
"""

import re
from typing import List

import pandas as pd


def tokenize_text(text: str) -> List[str]:
    """
    Split text into individual tokens (words).

    Args:
        text (str): Clean text from text_cleaning module

    Returns:
        List[str]: List of tokens
    """
    if not text or pd.isna(text):
        return []

    # Simple tokenization using regex (more reliable than NLTK)
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def remove_stop_words(tokens: List[str]) -> List[str]:
    """
    Remove common stop words from tokens.

    Args:
        tokens (List[str]): List of tokens

    Returns:
        List[str]: Tokens with stop words removed
    """
    if not tokens:
        return []

    # Common English stop words (no NLTK dependency)
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "would",
        "you",
        "your",
        "i",
        "me",
        "my",
        "we",
        "our",
        "they",
        "them",
        "their",
        "this",
        "these",
        "those",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "shall",
        "am",
        "been",
        "being",
        "was",
        "were",
        "or",
        "but",
        "if",
        "then",
        "because",
        "so",
        "than",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        "also",
        "well",
        "back",
        "even",
        "still",
        "yet",
    }

    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Convert tokens to their root forms using simple stemming.

    Args:
        tokens (List[str]): List of tokens

    Returns:
        List[str]: Stemmed tokens
    """
    if not tokens:
        return []

    # Simple stemming rules (no NLTK dependency)
    stemmed_tokens = []
    for token in tokens:
        original_token = token

        # Remove common suffixes
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("er") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("est") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ly") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]

        # If stemming made the word too short, keep original
        if len(token) < 3:
            token = original_token

        stemmed_tokens.append(token)

    return stemmed_tokens


def preprocess_text_pipeline(text: str) -> str:
    """
    Complete NLP preprocessing pipeline: tokenize -> remove stop words -> lemmatize -> join.

    Args:
        text (str): Clean text from text_cleaning module

    Returns:
        str: Fully processed text ready for NLP algorithms
    """
    if not text or pd.isna(text):
        return ""

    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens)
    tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


def preprocess_articles_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply NLP preprocessing to entire DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'cleaned_text' column

    Returns:
        pd.DataFrame: DataFrame with additional NLP columns
    """
    if df.empty or "cleaned_text" not in df.columns:
        return df

    processed_df = df.copy()

    processed_df["tokens"] = processed_df["cleaned_text"].apply(tokenize_text)
    processed_df["tokens_no_stop"] = processed_df["tokens"].apply(remove_stop_words)
    processed_df["tokens_lemmatized"] = processed_df["tokens_no_stop"].apply(
        lemmatize_tokens
    )
    processed_df["processed_text"] = processed_df["cleaned_text"].apply(
        preprocess_text_pipeline
    )

    processed_df["token_count"] = processed_df["tokens"].apply(len)
    processed_df["token_count_no_stop"] = processed_df["tokens_no_stop"].apply(len)
    processed_df["token_count_lemmatized"] = processed_df["tokens_lemmatized"].apply(
        len
    )

    return processed_df


def get_nlp_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about NLP processing.

    Args:
        df (pd.DataFrame): DataFrame with NLP columns

    Returns:
        dict: NLP processing statistics
    """
    if df.empty or "tokens" not in df.columns:
        return {}

    stats = {
        "total_articles": len(df),
        "average_tokens_per_article": (
            df["token_count"].mean() if "token_count" in df.columns else 0
        ),
        "average_tokens_no_stop": (
            df["token_count_no_stop"].mean()
            if "token_count_no_stop" in df.columns
            else 0
        ),
        "average_tokens_lemmatized": (
            df["token_count_lemmatized"].mean()
            if "token_count_lemmatized" in df.columns
            else 0
        ),
        "stop_words_removed_percentage": 0,
    }

    if "token_count" in df.columns and "token_count_no_stop" in df.columns:
        total_tokens = df["token_count"].sum()
        tokens_no_stop = df["token_count_no_stop"].sum()
        if total_tokens > 0:
            stats["stop_words_removed_percentage"] = (
                (total_tokens - tokens_no_stop) / total_tokens
            ) * 100

    return stats
