"""
TextRank summarization algorithm implementation.

This module provides functions to summarize text using the TextRank algorithm,
which builds a graph of sentences and uses PageRank to identify important sentences.
"""

import re
from typing import List

import networkx as nx
import pandas as pd

try:
    from utils_nlp import preprocess_text_pipeline
except ImportError:
    from src.utils_nlp import preprocess_text_pipeline


def split_into_sentence(text: str) -> List[str]:
    """
    Split the text into sentences.
    """
    if not text:
        return []
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def calculate_sentence_similarity(sent1: str, sent2: str) -> float:
    """
    Calculate similarity between two sentences using Jaccard similarity.
    Uses our existing NLP pipeline for preprocessing.
    """
    if not sent1 or not sent2 or pd.isna(sent1) or pd.isna(sent2):
        return 0.0

    processed1 = preprocess_text_pipeline(sent1)
    processed2 = preprocess_text_pipeline(sent2)

    words1 = set(processed1.split())
    words2 = set(processed2.split())

    words1 = {w for w in words1 if w.strip()}
    words2 = {w for w in words2 if w.strip()}

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def build_sentence_graph(text: str) -> nx.Graph:
    """
    Build a similarity graph between sentences using NetworkX.

    Args:
        text (str): Input text to process

    Returns:
        networkx.Graph: Graph with sentences as nodes and similarities as edges
    """
    sentences = split_into_sentence(text)

    if len(sentences) < 2:
        return nx.Graph()

    graph = nx.Graph()

    for i, sentence in enumerate(sentences):
        graph.add_node(i, sentence=sentence)

    for i, sent_i in enumerate(sentences):
        for j in range(i + 1, len(sentences)):
            similarity = calculate_sentence_similarity(sent_i, sentences[j])
            if similarity > 0.1:
                graph.add_edge(i, j, weight=similarity)

    return graph


def filter_and_score_sentences(sentences: List[str]) -> List[tuple]:
    """
    Filter and score sentences for better readability and quality.

    Args:
        sentences (List[str]): List of sentences to filter and score

    Returns:
        List[tuple]: List of (sentence, score) tuples sorted by quality
    """
    filtered_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        # Skip empty sentences
        if not sentence:
            continue

        # Skip very short sentences (less than 10 words)
        if len(sentence.split()) < 10:
            continue

        # Skip sentences that are mostly timestamps or updates
        if re.match(r"^\d+\.\d+pm?\s+BST", sentence) or re.match(
            r"^Updated at \d+\.\d+pm?\s+BST", sentence
        ):
            continue

        # Skip sentences that are mostly numbers and timestamps
        if (
            len(re.findall(r"\d+\.\d+pm?\s+BST", sentence)) > 0
            and len(sentence.split()) < 20
        ):
            continue

        # Skip sentences that are just commentary fragments
        if sentence.startswith("Here's") and len(sentence.split()) < 15:
            continue

        # Score sentence quality
        score = 0

        # Prefer longer sentences (more informative)
        word_count = len(sentence.split())
        score += min(word_count * 0.1, 5)  # Max 5 points for length

        # Prefer sentences with proper capitalization
        if sentence[0].isupper():
            score += 1

        # Prefer sentences with proper ending punctuation
        if sentence.endswith((".", "!", "?")):
            score += 1

        # Penalize sentences with too many timestamps
        timestamp_count = len(re.findall(r"\d+\.\d+pm?\s+BST", sentence))
        score -= timestamp_count * 0.5

        # Prefer sentences that don't start with fragments
        if not sentence.startswith(("Here's", "Updated at", "Judges'", "Best dance")):
            score += 1

        filtered_sentences.append((sentence, score))

    # Sort by score (highest first)
    filtered_sentences.sort(key=lambda x: x[1], reverse=True)

    return filtered_sentences


def textrank_summarize(text: str, num_sentences: int = 3) -> str:
    """
    Summarize text using TextRank algorithm with improved readability.

    Args:
        text (str): Input text to summarize
        num_sentences (int): Number of sentences in summary (default: 3)

    Returns:
        str: Summarized text
    """
    if not text or pd.isna(text):
        return ""

    sentences = split_into_sentence(text)

    # Filter and score sentences for better quality
    filtered_sentences = filter_and_score_sentences(sentences)

    if not filtered_sentences:
        # If no good sentences found, return first few sentences
        return " ".join(sentences[: min(3, len(sentences))])

    # If we have fewer good sentences than requested, return what we have
    if len(filtered_sentences) <= num_sentences:
        return " ".join([sent for sent, score in filtered_sentences])

    # Use TextRank on the filtered sentences
    filtered_text = " ".join([sent for sent, score in filtered_sentences])
    filtered_sentences_list = [sent for sent, score in filtered_sentences]

    graph = build_sentence_graph(filtered_text)

    if graph.number_of_nodes() == 0:
        return " ".join([sent for sent, score in filtered_sentences[:num_sentences]])

    try:
        scores = nx.pagerank(graph, weight="weight")
    except nx.NetworkXError:
        return " ".join([sent for sent, score in filtered_sentences[:num_sentences]])

    ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_indices = [idx for idx, score in ranked_sentences[:num_sentences]]
    top_indices.sort()

    summary_sentences = [filtered_sentences_list[i] for i in top_indices]

    return " ".join(summary_sentences)


def summarize_articles(
    df: pd.DataFrame, summary_percentage: float = 0.1
) -> pd.DataFrame:
    """
    Apply TextRank summarization to all articles in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with processed articles
        summary_percentage (float): Percentage of original text length for summary (default: 0.1 = 10%)

    Returns:
        pd.DataFrame: DataFrame with added summary columns
    """
    if df.empty:
        return df

    result_df = df.copy()

    result_df["summary"] = ""
    result_df["summary_length"] = 0
    result_df["compression_ratio"] = 0.0

    for idx, row in result_df.iterrows():
        try:
            text_to_summarize = row.get("Content", "")

            if text_to_summarize and not pd.isna(text_to_summarize):
                # Calculate target summary length based on percentage
                original_length = len(text_to_summarize.split())
                target_summary_length = max(
                    1, int(original_length * summary_percentage)
                )

                # Split into sentences to determine how many sentences we need
                sentences = split_into_sentence(text_to_summarize)

                if len(sentences) <= 1:
                    # If only one sentence, return it as is
                    summary = text_to_summarize
                else:
                    # Calculate approximate sentences needed based on average sentence length
                    avg_sentence_length = original_length / len(sentences)
                    num_sentences_needed = max(
                        1, int(target_summary_length / avg_sentence_length)
                    )

                    # Don't exceed the total number of sentences
                    num_sentences_needed = min(num_sentences_needed, len(sentences))

                    summary = textrank_summarize(
                        text_to_summarize, num_sentences_needed
                    )

                result_df.at[idx, "summary"] = summary
                result_df.at[idx, "summary_length"] = len(summary.split())

                if original_length > 0:
                    result_df.at[idx, "compression_ratio"] = (
                        len(summary.split()) / original_length
                    )
            else:
                result_df.at[idx, "summary"] = ""
                result_df.at[idx, "summary_length"] = 0
                result_df.at[idx, "compression_ratio"] = 0.0

        except Exception as e:
            result_df.at[idx, "summary"] = f"Error: {str(e)}"
            result_df.at[idx, "summary_length"] = 0
            result_df.at[idx, "compression_ratio"] = 0.0

    return result_df


def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about the summaries in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with summary columns

    Returns:
        dict: Dictionary with summary statistics
    """
    if df.empty or "summary" not in df.columns:
        return {}

    stats = {}

    stats["total_articles"] = len(df)
    stats["articles_with_summaries"] = len(df[df["summary"].str.len() > 0])

    if "summary_length" in df.columns:
        summary_lengths = df[df["summary_length"] > 0]["summary_length"].dropna()
        if len(summary_lengths) > 0:
            stats["avg_summary_length"] = summary_lengths.mean()
            stats["min_summary_length"] = summary_lengths.min()
            stats["max_summary_length"] = summary_lengths.max()

    if "compression_ratio" in df.columns:
        compression_ratios = df["compression_ratio"].dropna()
        if len(compression_ratios) > 0:
            stats["avg_compression_ratio"] = compression_ratios.mean()
            stats["min_compression_ratio"] = compression_ratios.min()
            stats["max_compression_ratio"] = compression_ratios.max()

    return stats
