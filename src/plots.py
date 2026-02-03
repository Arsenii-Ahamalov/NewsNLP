"""
Plotting functions for data visualization.

This module provides functions to create various plots for topic modeling
and article analysis, including distribution charts, scatter plots, heatmaps,
and word clouds.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from wordcloud import WordCloud

# Set style for better looking plots
plt.style.use("default")
sns.set_palette("husl")


def plot_topic_distribution(
    df: pd.DataFrame,
    title: str = "Topic Distribution",
    topic_to_name: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Create a bar chart showing the distribution of articles across topics.

    Args:
        df: DataFrame with 'Topic' column
        title: Plot title
        topic_to_name: Optional mapping from topic number to topic name

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set larger font sizes for better visibility
    plt.rcParams.update({"font.size": 12})
    ax.tick_params(labelsize=11)

    # Count articles per topic
    topic_counts = df["Topic"].value_counts().sort_index()

    # Create bar plot
    bars = ax.bar(
        range(len(topic_counts)),
        topic_counts.values,
        color=sns.color_palette("husl", len(topic_counts)),
    )

    # Use topic names if provided, otherwise use numbers
    if topic_to_name:
        topic_labels = [topic_to_name.get(i, f"Topic {i}") for i in topic_counts.index]
    else:
        topic_labels = [f"Topic {i}" for i in topic_counts.index]

    # Customize plot
    ax.set_xlabel("Topic", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Articles", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(range(len(topic_counts)))
    ax.set_xticklabels(topic_labels, rotation=45, ha="right", fontsize=11)

    # Add value labels on bars
    for bar_item, count in zip(bars, topic_counts.values):
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2,
            bar_item.get_height() + 0.1,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Add percentage labels
    total_articles = len(df)
    for i, count in enumerate(topic_counts.values):
        percentage = (count / total_articles) * 100
        ax.text(
            i,
            count / 2,
            f"{percentage:.1f}%",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
            fontsize=11,
        )

    plt.tight_layout()
    return fig


def plot_top_keywords_per_topic(
    top_keywords: List[List[str]],
    top_n: int = 10,
    title: str = "Top Keywords per Topic",
    topic_to_name: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Create bar charts showing top keywords for each topic.

    Args:
        top_keywords: List of keyword lists, one per topic
        top_n: Number of top keywords to show per topic
        title: Plot title
        topic_to_name: Optional mapping from topic number to topic name

    Returns:
        matplotlib Figure object
    """
    n_topics = len(top_keywords)
    fig, axes = plt.subplots(1, n_topics, figsize=(5 * n_topics, 6))

    # Set larger font sizes for better visibility
    plt.rcParams.update({"font.size": 14})

    if n_topics == 1:
        axes = [axes]

    for i, keywords in enumerate(top_keywords):
        ax = axes[i]

        # Take top N keywords
        top_kw = keywords[:top_n]

        # Create horizontal bar plot
        y_pos = np.arange(len(top_kw))
        ax.barh(
            y_pos,
            range(len(top_kw), 0, -1),
            color=sns.color_palette("husl", n_topics)[i],
        )

        # Use topic name if provided
        topic_label = (
            topic_to_name.get(i, f"Topic {i}") if topic_to_name else f"Topic {i}"
        )

        # Customize with larger fonts for better visibility
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_kw, fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=16, fontweight="bold")
        ax.set_title(topic_label, fontsize=18, fontweight="bold", pad=15)
        ax.tick_params(labelsize=12, width=1.5, length=5)
        ax.invert_yaxis()  # Most important at top

        # Remove duplicate keyword labels (already shown on y-axis)
        # The y-axis labels are sufficient

    plt.suptitle(title, fontsize=22, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_topic_scatter(
    tfidf_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],  # noqa: ARG001
    title: str = "Topic Clusters (2D PCA)",
    topic_to_name: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Create a 2D scatter plot of articles colored by topic using PCA.

    Args:
        tfidf_matrix: TF-IDF matrix
        labels: Topic labels for each article
        feature_names: Feature names (for potential hover info)
        title: Plot title
        topic_to_name: Optional mapping from topic number to topic name

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set larger font sizes for better visibility
    plt.rcParams.update({"font.size": 11})

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    tfidf_2d = pca.fit_transform(tfidf_matrix)

    # Get unique topics and colors
    unique_topics = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_topics))

    # Plot each topic
    for i, topic in enumerate(unique_topics):
        mask = labels == topic
        # Use topic name if provided
        topic_label = (
            topic_to_name.get(int(topic), f"Topic {topic}")
            if topic_to_name
            else f"Topic {topic}"
        )
        ax.scatter(
            tfidf_2d[mask, 0],
            tfidf_2d[mask, 1],
            c=[colors[i]],
            label=topic_label,
            alpha=0.7,
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )

    # Customize plot
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    # Add total variance explained
    total_variance = pca.explained_variance_ratio_.sum()
    ax.text(
        0.02,
        0.98,
        f"Total variance explained: {total_variance:.1%}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "black"},
    )

    plt.tight_layout()
    return fig


def plot_keywords_heatmap(
    top_keywords: List[List[str]],
    top_n: int = 10,
    title: str = "Keywords Importance Heatmap",
    topic_to_name: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Create a heatmap showing keyword importance across topics.

    Args:
        top_keywords: List of keyword lists, one per topic
        top_n: Number of top keywords to include
        title: Plot title
        topic_to_name: Optional mapping from topic number to topic name

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set larger font sizes for better visibility
    plt.rcParams.update({"font.size": 10})

    # Create matrix: topics x keywords
    n_topics = len(top_keywords)

    # Get all unique keywords from top N of each topic
    all_keywords = set()
    for keywords in top_keywords:
        all_keywords.update(keywords[:top_n])
    all_keywords = sorted(list(all_keywords))

    # Create importance matrix
    importance_matrix = np.zeros((n_topics, len(all_keywords)))

    for topic_idx, keywords in enumerate(top_keywords):
        for rank, keyword in enumerate(keywords[:top_n]):
            if keyword in all_keywords:
                col_idx = all_keywords.index(keyword)
                # Higher rank = higher importance (lower number = higher importance)
                importance_matrix[topic_idx, col_idx] = top_n - rank

    # Use topic names if provided
    if topic_to_name:
        topic_labels = [topic_to_name.get(i, f"Topic {i}") for i in range(n_topics)]
    else:
        topic_labels = [f"Topic {i}" for i in range(n_topics)]

    # Create heatmap
    sns.heatmap(
        importance_matrix,
        xticklabels=all_keywords,
        yticklabels=topic_labels,
        cmap="YlOrRd",
        cbar_kws={"label": "Importance Score", "shrink": 0.8},
        ax=ax,
        annot=False,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Keywords", fontsize=12, fontweight="bold")
    ax.set_ylabel("Topics", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig


def create_wordclouds(
    top_keywords: List[List[str]],
    titles: Optional[List[str]] = None,
    topic_to_name: Optional[Dict[int, str]] = None,
) -> List[plt.Figure]:
    """
    Create word clouds for each topic.

    Args:
        top_keywords: List of keyword lists, one per topic
        titles: Optional list of titles for each topic
        topic_to_name: Optional mapping from topic number to topic name

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    for i, keywords in enumerate(top_keywords):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create word cloud
        if keywords:
            # Create frequency dict (higher rank = higher frequency)
            word_freq = {}
            for rank, word in enumerate(keywords):
                word_freq[word] = len(keywords) - rank

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="viridis",
                max_words=50,
            ).generate_from_frequencies(word_freq)

            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")

            # Use topic name if provided, otherwise use title or default
            if topic_to_name:
                title = topic_to_name.get(
                    i, titles[i] if titles and i < len(titles) else f"Topic {i}"
                )
            else:
                title = titles[i] if titles and i < len(titles) else f"Topic {i}"
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        else:
            topic_label = (
                topic_to_name.get(i, f"Topic {i}") if topic_to_name else f"Topic {i}"
            )
            ax.text(
                0.5,
                0.5,
                "No keywords available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(f"{topic_label} (No Keywords)", fontsize=16, fontweight="bold")

        figures.append(fig)

    return figures


def plot_topic_analysis_summary(
    df: pd.DataFrame,
    top_keywords: List[List[str]],
    tfidf_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],  # noqa: ARG001
) -> plt.Figure:
    """
    Create a comprehensive summary plot with multiple subplots.

    Args:
        df: DataFrame with articles and topics
        top_keywords: List of keyword lists per topic
        tfidf_matrix: TF-IDF matrix
        labels: Topic labels
        feature_names: Feature names

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12))

    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # 1. Topic distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    topic_counts = df["Topic"].value_counts().sort_index()
    ax1.bar(
        range(len(topic_counts)),
        topic_counts.values,
        color=sns.color_palette("husl", len(topic_counts)),
    )
    ax1.set_title("Topic Distribution")
    ax1.set_xlabel("Topic")
    ax1.set_ylabel("Number of Articles")
    ax1.set_xticks(range(len(topic_counts)))
    ax1.set_xticklabels([f"T{i}" for i in topic_counts.index])

    # Add percentages
    total = len(df)
    for i, count in enumerate(topic_counts.values):
        ax1.text(
            i,
            count + 0.1,
            f"{count/total*100:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Top keywords for first topic (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if top_keywords:
        first_topic_kw = top_keywords[0][:8]  # Top 8 keywords
        y_pos = np.arange(len(first_topic_kw))
        ax2.barh(y_pos, range(len(first_topic_kw), 0, -1))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(first_topic_kw)
        ax2.set_title("Top Keywords - Topic 0")
        ax2.invert_yaxis()

    # 3. PCA scatter plot (bottom left)
    ax3 = fig.add_subplot(gs[1, :])
    pca = PCA(n_components=2, random_state=42)
    tfidf_2d = pca.fit_transform(tfidf_matrix)

    unique_topics = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_topics))

    for i, topic in enumerate(unique_topics):
        mask = labels == topic
        ax3.scatter(
            tfidf_2d[mask, 0],
            tfidf_2d[mask, 1],
            c=[colors[i]],
            label=f"Topic {topic}",
            alpha=0.7,
            s=100,
        )

    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax3.set_title("Topic Clusters (2D PCA)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Keywords summary (bottom right)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")

    # Create text summary
    summary_text = "Topic Analysis Summary:\n\n"
    for i, keywords in enumerate(top_keywords):
        summary_text += f"Topic {i}: {', '.join(keywords[:5])}\n"

    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
    )

    plt.suptitle("Topic Modeling Analysis Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def get_topic_names_from_sections(df: pd.DataFrame, metadata: Dict) -> Dict[int, str]:
    """
    Generate topic names from most common sections in articles.

    Args:
        df: DataFrame with articles and Topic column
        metadata: Metadata dictionary with topic information

    Returns:
        Dictionary mapping topic number to topic name
    """
    topic_to_name = {}

    for i in range(metadata["parameters"]["best_k"]):
        topic_articles = df[df["Topic"] == i]

        if len(topic_articles) > 0:
            section_counts = topic_articles["Section"].value_counts()
            if len(section_counts) > 0:
                most_common_section = section_counts.index[0]
                topic_to_name[i] = most_common_section
            else:
                topic_to_name[i] = f"Topic {i}"
        else:
            topic_to_name[i] = f"Topic {i}"

    return topic_to_name


def save_all_plots(
    df: pd.DataFrame,
    top_keywords: List[List[str]],
    tfidf_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    output_dir: str = "data/processed/plots/",
    topic_to_name: Optional[Dict[int, str]] = None,
) -> Dict[str, str]:
    """
    Save all visualization plots to files.

    Args:
        df: DataFrame with articles and topics
        top_keywords: List of keyword lists per topic
        tfidf_matrix: TF-IDF matrix
        labels: Topic labels
        feature_names: Feature names
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}

    # Load metadata if topic_to_name not provided
    if topic_to_name is None:
        try:
            with open("data/processed/topic_modeling_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
            topic_to_name = get_topic_names_from_sections(df, metadata)
        except FileNotFoundError:
            topic_to_name = None

    # 1. Topic distribution
    fig1 = plot_topic_distribution(df, topic_to_name=topic_to_name)
    file1 = os.path.join(output_dir, "topic_distribution.png")
    fig1.savefig(file1, dpi=300, bbox_inches="tight")
    saved_files["topic_distribution"] = file1
    plt.close(fig1)

    # 2. Top keywords per topic
    fig2 = plot_top_keywords_per_topic(top_keywords, topic_to_name=topic_to_name)
    file2 = os.path.join(output_dir, "top_keywords_per_topic.png")
    fig2.savefig(file2, dpi=300, bbox_inches="tight")
    saved_files["top_keywords"] = file2
    plt.close(fig2)

    # 3. Topic scatter plot
    fig3 = plot_topic_scatter(
        tfidf_matrix, labels, feature_names, topic_to_name=topic_to_name
    )
    file3 = os.path.join(output_dir, "topic_scatter.png")
    fig3.savefig(file3, dpi=300, bbox_inches="tight")
    saved_files["topic_scatter"] = file3
    plt.close(fig3)

    # 4. Keywords heatmap
    fig4 = plot_keywords_heatmap(top_keywords, topic_to_name=topic_to_name)
    file4 = os.path.join(output_dir, "keywords_heatmap.png")
    fig4.savefig(file4, dpi=300, bbox_inches="tight")
    saved_files["keywords_heatmap"] = file4
    plt.close(fig4)

    # 5. Word clouds
    wordcloud_figs = create_wordclouds(top_keywords, topic_to_name=topic_to_name)
    for i, fig in enumerate(wordcloud_figs):
        file = os.path.join(output_dir, f"wordcloud_topic_{i}.png")
        fig.savefig(file, dpi=300, bbox_inches="tight")
        saved_files[f"wordcloud_topic_{i}"] = file
        plt.close(fig)

    # 6. Summary plot
    fig_summary = plot_topic_analysis_summary(
        df, top_keywords, tfidf_matrix, labels, feature_names
    )
    file_summary = os.path.join(output_dir, "topic_analysis_summary.png")
    fig_summary.savefig(file_summary, dpi=300, bbox_inches="tight")
    saved_files["summary"] = file_summary
    plt.close(fig_summary)

    return saved_files
