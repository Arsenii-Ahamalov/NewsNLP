"""
Streamlit dashboard for Smart News Summarizer & Analyzer.

This module provides an interactive web interface for viewing and managing
news articles, topics, summaries, and visualizations.
"""

import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# Set page config
st.set_page_config(
    page_title="Smart News Summarizer & Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .topic-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .article-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #28a745;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_data():
    """Load all processed data files."""
    try:
        # Load articles with topics
        articles_df = pd.read_csv("data/processed/articles_with_topics.csv")

        # Remove duplicates by title (hotfix - keep first occurrence)
        # Normalize titles for comparison (strip whitespace, lowercase)
        if "Title" in articles_df.columns:
            articles_df = articles_df.drop_duplicates(
                subset="Title", keep="first"
            ).reset_index(drop=True)

        # Ensure summary_length exists and is calculated from summary if missing
        if "summary" in articles_df.columns:
            # Fill missing summary_length by calculating from summary
            mask = articles_df["summary"].notna() & articles_df["summary"].astype(str).str.strip().str.len() > 0
            if "summary_length" not in articles_df.columns:
                articles_df["summary_length"] = 0
            # Recalculate summary_length for rows where it's missing or 0 but summary exists
            recalc_mask = mask & (
                articles_df["summary_length"].isna()
                | (articles_df["summary_length"] == 0)
            )
            if recalc_mask.any():
                articles_df.loc[recalc_mask, "summary_length"] = (
                    articles_df.loc[recalc_mask, "summary"]
                    .astype(str)
                    .str.split()
                    .str.len()
                )

        # Load metadata
        with open(
            "data/processed/topic_modeling_metadata.json", "r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        return articles_df, metadata
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please run the data processing pipeline first.")
        return None, None


def get_topic_names_from_sections(articles_df, metadata):
    """
    Generate human-readable topic names from most common sections.
    Groups topics with the same common section together.
    Returns:
        tuple: (topic_to_name dict, name_to_topics dict, grouped_topic_names list)
        - topic_to_name: {0: "US news", 1: "US news", ...}
        - name_to_topics: {"US news": [0, 1], "Football": [2], ...}
        - grouped_topic_names: ["US news", "Football", ...] for dropdown
    """
    topic_to_name = {}
    section_to_topics = {}
    # Find most common section for each topic
    for i in range(metadata["parameters"]["best_k"]):
        topic_articles = articles_df[articles_df["Topic"] == i]

        if len(topic_articles) > 0:
            # Get most common section
            section_counts = topic_articles["Section"].value_counts()
            if len(section_counts) > 0:
                most_common_section = section_counts.index[0]
                topic_to_name[i] = most_common_section
                # Group topics by common section
                if most_common_section not in section_to_topics:
                    section_to_topics[most_common_section] = []
                section_to_topics[most_common_section].append(i)
            else:
                topic_to_name[i] = f"Topic {i}"
                if f"Topic {i}" not in section_to_topics:
                    section_to_topics[f"Topic {i}"] = []
                section_to_topics[f"Topic {i}"].append(i)
        else:
            topic_to_name[i] = f"Topic {i}"
            if f"Topic {i}" not in section_to_topics:
                section_to_topics[f"Topic {i}"] = []
            section_to_topics[f"Topic {i}"].append(i)
    # Create reverse mapping: section name -> list of topic numbers
    name_to_topics = section_to_topics.copy()
    # Create list of unique section names for dropdown (sorted by number of topics)
    grouped_topic_names = sorted(
        name_to_topics.keys(),
        key=lambda x: (len(name_to_topics[x]), x),
        reverse=True
    )
    return topic_to_name, name_to_topics, grouped_topic_names


def create_wordcloud(text, title):
    """Create a word cloud from text."""
    if not text or pd.isna(text):
        return None

    # Convert text to string and clean
    text = str(text)

    # Create word cloud
    wordcloud = WordCloud(
        width=400,
        height=300,
        background_color="white",
        colormap="viridis",
        max_words=50,
        relative_scaling=0.5,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    return fig


def plot_topic_distribution(df, topic_to_name=None):
    """Create topic distribution plot."""
    topic_counts = df["Topic"].value_counts().sort_index()
    # Use topic names if provided, otherwise use numbers
    if topic_to_name:
        topic_labels = [topic_to_name.get(i, f"Topic {i}") for i in topic_counts.index]
    else:
        topic_labels = [f"Topic {i}" for i in topic_counts.index]

    fig = px.bar(
        x=topic_labels,
        y=topic_counts.values,
        title="Article Distribution by Topic",
        labels={"x": "Topic", "y": "Number of Articles"},
        color=topic_counts.values,
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Topic",
        yaxis_title="Number of Articles",
    )

    return fig


def plot_topic_keywords(metadata, topic_to_name=None):
    """Create topic keywords visualization."""
    topics_data = []
    for topic_id, topic_info in metadata["topics"].items():
        # Extract topic number from topic_id (e.g., "topic_0" -> 0)
        topic_num = int(topic_id.split("_")[1]) if "_" in topic_id else int(topic_id.replace("topic", ""))
        # Use topic name if provided
        topic_label = topic_to_name.get(topic_num, topic_id) if topic_to_name else topic_id
        for keyword in topic_info["keywords"][:5]:  # Top 5 keywords
            topics_data.append(
                {
                    "Topic": topic_label,
                    "Keyword": keyword,
                    "Count": topic_info["article_count"],
                }
            )

    if not topics_data:
        return None

    df_keywords = pd.DataFrame(topics_data)

    fig = px.bar(
        df_keywords,
        x="Topic",
        y="Count",
        color="Keyword",
        title="Top Keywords by Topic",
        height=400,
    )

    return fig


def main():
    """Main dashboard function."""

    # Header
    st.markdown(
        '<h1 class="main-header">📰 Smart News Summarizer & Analyzer</h1>',
        unsafe_allow_html=True,
    )

    # Load data (always read latest CSV/metadata on rerun)
    articles_df, metadata = load_data()

    if articles_df is None or metadata is None:
        st.stop()

    # Generate topic names from sections and group them
    topic_to_name, name_to_topics, grouped_topic_names = get_topic_names_from_sections(
        articles_df, metadata
    )

    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")

    # Data overview metrics
    st.sidebar.markdown("### 📊 Data Overview")
    st.sidebar.metric("Total Articles", len(articles_df))
    st.sidebar.metric("Topics Discovered", metadata["parameters"]["best_k"])
    st.sidebar.metric(
        "Silhouette Score", f"{metadata['parameters']['silhouette_score']:.3f}"
    )

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🏠 Overview",
            "📰 Articles by Topic",
            "📊 Visualizations",
            "🔍 Search & Filter",
            "📥 Download Data",
            "🔄 Process New Articles",
        ]
    )

    with tab1:
        st.markdown("## 🏠 Dashboard Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Articles", len(articles_df), help="Number of articles processed"
            )

        with col2:
            st.metric(
                "Topics Found",
                metadata["parameters"]["best_k"],
                help="Number of topics discovered by clustering",
            )

        with col3:
            st.metric(
                "Avg Articles/Topic",
                f"{len(articles_df) / metadata['parameters']['best_k']:.1f}",
                help="Average articles per topic",
            )

        with col4:
            st.metric(
                "Clustering Quality",
                f"{metadata['parameters']['silhouette_score']:.3f}",
                help="Silhouette score (higher = better clustering)",
            )

        # Topic distribution
        st.markdown("### 📊 Topic Distribution")
        fig_dist_overview = plot_topic_distribution(articles_df, topic_to_name)
        st.plotly_chart(
            fig_dist_overview, use_container_width=True, key="overview_topic_dist"
        )

        # Recent articles preview
        st.markdown("### 📰 Recent Articles Preview")
        recent_articles = articles_df[
            ["Title", "Section", "Topic", "Author", "Published Date"]
        ].head(5)
        st.dataframe(recent_articles, use_container_width=True)

    with tab2:
        st.markdown("## 📰 Articles by Topic")

        # Topic selector (using grouped section names)
        selected_topic_name = st.selectbox("Select a topic to explore:", grouped_topic_names)

        if selected_topic_name:
            # Get all topic numbers for this section name
            topic_nums = name_to_topics[selected_topic_name]

            # Filter articles for all topics with this section name
            topic_articles = articles_df[articles_df["Topic"].isin(topic_nums)]

            if len(topic_articles) > 0:
                # Topic info
                st.markdown(f"### 🎯 {selected_topic_name}")
                # Show which original topics are included if multiple
                if len(topic_nums) > 1:
                    st.caption(f"Combined from topics: {', '.join([f'Topic {t}' for t in topic_nums])}")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Articles in this topic:** {len(topic_articles)}")
                    common_sections = (
                        topic_articles["Section"].value_counts().head(3).index.tolist()
                    )
                    st.markdown("**Common sections:** " + ", ".join(common_sections))

                with col2:
                    # Topic keywords (combine from all topics in this group)
                    all_keywords = []
                    for topic_num in topic_nums:
                        topic_key = f"topic_{topic_num}"
                        if topic_key in metadata["topics"]:
                            all_keywords.extend(metadata["topics"][topic_key]["keywords"][:5])
                    # Get unique keywords, limit to top 10
                    unique_keywords = list(dict.fromkeys(all_keywords))[:10]
                    st.markdown("**Top Keywords:**")
                    for keyword in unique_keywords:
                        st.markdown(f"• {keyword}")

                # Articles in this topic
                st.markdown("### 📄 Articles")

                for idx, article in topic_articles.iterrows():
                    with st.expander(f"📰 {article['Title']}"):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**Section:** {article['Section']}")
                            st.markdown(f"**Author:** {article['Author']}")
                            st.markdown(f"**Published:** {article['Published Date']}")
                            st.markdown(
                                f"**URL:** [Read full article]({article['Url']})"
                            )

                        with col2:
                            st.markdown(f"**Word Count:** {article['word_count']}")
                            # Calculate summary length from summary if available
                            if (
                                "summary" in article
                                and pd.notna(article["summary"])
                                and str(article["summary"]).strip()
                            ):
                                summary_length = len(str(article["summary"]).split())
                                st.markdown(f"**Summary Length:** {summary_length} words")
                            elif (
                                "summary_length" in article
                                and pd.notna(article["summary_length"])
                                and article["summary_length"] > 0
                            ):
                                st.markdown(
                                    f"**Summary Length:** "
                                    f"{int(article['summary_length'])} words"
                                )
                            else:
                                st.markdown("**Summary Length:** No summary")

                        # Summary
                        if "summary" in article and pd.notna(article["summary"]):
                            st.markdown("**📝 Summary:**")
                            st.markdown(
                                article["summary"][:500] + "..."
                                if len(str(article["summary"])) > 500
                                else article["summary"]
                            )
                        else:
                            st.markdown("**📝 Content Preview:**")
                            st.markdown(
                                article["Content"][:500] + "..."
                                if len(article["Content"]) > 500
                                else article["Content"]
                            )
            else:
                st.warning(f"No articles found for {selected_topic_name}")

    with tab3:
        st.markdown("## 📊 Visualizations")

        # Topic distribution
        st.markdown("### 📊 Topic Distribution")
        fig_dist_viz = plot_topic_distribution(articles_df, topic_to_name)
        st.plotly_chart(
            fig_dist_viz, use_container_width=True, key="visualizations_topic_dist"
        )

        # Topic keywords
        st.markdown("### 🔤 Topic Keywords")
        fig_keywords = plot_topic_keywords(metadata, topic_to_name)
        if fig_keywords:
            st.plotly_chart(
                fig_keywords,
                use_container_width=True,
                key="visualizations_topic_keywords",
            )

        # Word clouds for each (grouped) topic
        st.markdown("### ☁️ Word Clouds by Topic")

        # Group topics that share the same human-readable name (e.g., both mapped to 'Sport')
        label_to_texts: dict[str, list[str]] = {}

        for topic_num in range(metadata["parameters"]["best_k"]):
            topic_articles = articles_df[articles_df["Topic"] == topic_num]
            if len(topic_articles) == 0:
                continue

            # Combine all processed text for this topic
            topic_text = " ".join(
                topic_articles["processed_text"].dropna().astype(str)
            )
            if not topic_text:
                continue

            # Use grouped topic name if available, otherwise fall back to numeric label
            topic_label = topic_to_name.get(topic_num, f"Topic {topic_num}")
            label_to_texts.setdefault(topic_label, []).append(topic_text)

        # Limit to first N grouped topics for display
        max_wordclouds = 6
        cols = st.columns(2)

        for idx, (label, texts) in enumerate(list(label_to_texts.items())[:max_wordclouds]):
            combined_text = " ".join(texts)
            fig_wc = create_wordcloud(combined_text, label)
            if fig_wc:
                with cols[idx % 2]:
                    st.pyplot(fig_wc)

        # Section distribution
        st.markdown("### 📰 Articles by Section")
        section_counts = articles_df["Section"].value_counts()

        fig_sections = px.pie(
            values=section_counts.values,
            names=section_counts.index,
            title="Article Distribution by Section",
        )

        st.plotly_chart(
            fig_sections, use_container_width=True, key="visualizations_section_dist"
        )

    with tab4:
        st.markdown("## 🔍 Search & Filter")

        # Search functionality
        search_term = st.text_input("🔍 Search articles by title or content:", "")

        # Filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            section_filter = st.selectbox(
                "Filter by Section:", ["All"] + articles_df["Section"].unique().tolist()
            )

        with col2:
            # Topic filter (using grouped section names)
            topic_filter_options = ["All"] + grouped_topic_names
            topic_filter = st.selectbox(
                "Filter by Topic:",
                topic_filter_options,
            )

        with col3:
            min_words = st.slider(
                "Minimum word count:",
                min_value=0,
                max_value=int(articles_df["word_count"].max()),
                value=0,
            )

        # Apply filters
        filtered_df = articles_df.copy()

        if search_term:
            mask = filtered_df["Title"].str.contains(
                search_term, case=False, na=False
            ) | filtered_df["Content"].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]

        if section_filter != "All":
            filtered_df = filtered_df[filtered_df["Section"] == section_filter]

        if topic_filter != "All":
            # Get all topic numbers for this section name
            topic_nums = name_to_topics[topic_filter]
            filtered_df = filtered_df[filtered_df["Topic"].isin(topic_nums)]

        filtered_df = filtered_df[filtered_df["word_count"] >= min_words]

        # Display results
        st.markdown(f"### 📊 Found {len(filtered_df)} articles")

        if len(filtered_df) > 0:
            # Display as cards
            for idx, article in filtered_df.iterrows():
                topic_display_name = topic_to_name.get(article['Topic'], f"Topic {article['Topic']}")
                with st.expander(f"📰 {article['Title']} ({topic_display_name})"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Section:** {article['Section']}")
                        st.markdown(f"**Author:** {article['Author']}")
                        st.markdown(f"**Published:** {article['Published Date']}")
                        st.markdown(f"**Word Count:** {article['word_count']}")

                    with col2:
                        topic_display_name = topic_to_name.get(article['Topic'], f"Topic {article['Topic']}")
                        st.markdown(f"**Topic:** {topic_display_name}")
                        st.markdown(f"**URL:** [Read full article]({article['Url']})")

                    # Summary or content preview
                    if "summary" in article and pd.notna(article["summary"]):
                        st.markdown("**📝 Summary:**")
                        st.markdown(article["summary"])
                    else:
                        st.markdown("**📝 Content Preview:**")
                        st.markdown(
                            article["Content"][:300] + "..."
                            if len(article["Content"]) > 300
                            else article["Content"]
                        )
        else:
            st.info("No articles match your search criteria.")

    with tab5:
        st.markdown("## 📥 Download Data")

        # Download options
        st.markdown("### 📊 Download Processed Data")

        col1, col2 = st.columns(2)

        with col1:
            # Download articles with topics
            csv_data = articles_df.to_csv(index=False)
            st.download_button(
                label="📰 Download Articles with Topics (CSV)",
                data=csv_data,
                file_name=f"articles_with_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            # Download metadata
            metadata_json = json.dumps(metadata, indent=2)
            st.download_button(
                label="📋 Download Topic Modeling Metadata (JSON)",
                data=metadata_json,
                file_name=f"topic_modeling_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        with col2:
            # Download summary statistics
            summary_stats = {
                "total_articles": len(articles_df),
                "topics_discovered": metadata["parameters"]["best_k"],
                "silhouette_score": metadata["parameters"]["silhouette_score"],
                "avg_articles_per_topic": len(articles_df)
                / metadata["parameters"]["best_k"],
                "sections": articles_df["Section"].value_counts().to_dict(),
                "topic_distribution": articles_df["Topic"].value_counts().to_dict(),
            }

            summary_json = json.dumps(summary_stats, indent=2)
            st.download_button(
                label="📈 Download Summary Statistics (JSON)",
                data=summary_json,
                file_name=f"summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        # Data preview
        st.markdown("### 👀 Data Preview")

        preview_option = st.selectbox(
            "Select data to preview:",
            ["Articles with Topics", "Topic Metadata", "Summary Statistics"],
        )

        if preview_option == "Articles with Topics":
            st.dataframe(articles_df.head(10), use_container_width=True)
        elif preview_option == "Topic Metadata":
            st.json(metadata)
        elif preview_option == "Summary Statistics":
            st.json(summary_stats)

    with tab6:
        st.markdown("## 🔄 Process New Articles")
        st.markdown(
            "Download fresh articles from The Guardian API and add them to your database."
        )

        # Import processing functions
        try:
            from database_manager import (
                get_database_stats,
                regenerate_summaries_for_database,
                update_article_database,
            )
            from fetch_articles import get_articles
            from text_cleaning import preprocess_articles as clean_articles
            from textrank import summarize_articles
            from utils_nlp import preprocess_articles_nlp

            processing_available = True
        except ImportError as e:
            st.error(f"❌ Processing modules not available: {e}")
            processing_available = False

        if processing_available:
            # Database statistics - use same stats as Overview tab (hotfix)
            st.markdown("### 📊 Current Database Status")

            # Use a key to force refresh when needed
            refresh_key = st.session_state.get("db_refresh_key", 0)

            # Add refresh and regenerate buttons
            col_refresh, col_regenerate, _ = st.columns([1, 1, 3])
            with col_refresh:
                if st.button("🔄 Refresh Stats"):
                    st.session_state.db_refresh_key = refresh_key + 1
                    st.rerun()
            with col_regenerate:
                if st.button("📝 Regenerate Summaries"):
                    with st.spinner("🔄 Regenerating summaries for all articles..."):
                        success = regenerate_summaries_for_database()
                        if success:
                            st.success("✅ Summaries regenerated successfully!")
                            st.session_state.db_refresh_key = refresh_key + 1
                            st.rerun()
                        else:
                            st.error("❌ Failed to regenerate summaries")

            # Use same stats as Overview tab (hotfix)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Articles", len(articles_df))

            with col2:
                st.metric("Sections", len(articles_df["Section"].unique()))

            with col3:
                st.metric("Topics", metadata["parameters"]["best_k"])

            with col4:
                has_summaries = (
                    "summary" in articles_df.columns
                    and articles_df["summary"].notna().any()
                )
                st.metric("Has Summaries", "✅" if has_summaries else "❌")

            # Date range if available
            if "Published Date" in articles_df.columns:
                dates = pd.to_datetime(
                    articles_df["Published Date"], errors="coerce"
                ).dropna()
                if len(dates) > 0:
                    earliest = dates.min().strftime("%Y-%m-%d")
                    latest = dates.max().strftime("%Y-%m-%d")
                    st.info(f"📅 Date range: {earliest} to {latest}")

            # Article download parameters
            st.markdown("### 📥 Download Parameters")

            col1, col2 = st.columns(2)

            with col1:
                query = st.text_input(
                    "🔍 Search Query (optional)",
                    placeholder="e.g., 'artificial intelligence', 'climate change'",
                    help="Leave empty to get general news",
                )

                section = st.selectbox(
                    "📰 Section Filter (optional)",
                    [
                        "",
                        "world",
                        "politics",
                        "business",
                        "technology",
                        "science",
                        "sport",
                        "culture",
                        "lifestyle",
                        "environment",
                    ],
                    help="Leave empty to get articles from all sections",
                )

                number_of_articles = st.slider(
                    "📊 Number of Articles",
                    min_value=1,
                    max_value=50,
                    value=20,
                    help="Number of articles to download (1-50)",
                )

            with col2:
                from_date = st.date_input(
                    "📅 From Date (optional)",
                    value=None,
                    help="Start date for article search",
                )

                to_date = st.date_input(
                    "📅 To Date (optional)",
                    value=None,
                    help="End date for article search",
                )

                auto_process = st.checkbox(
                    "⚡ Auto-process after download",
                    value=True,
                    help="Automatically process, summarize, and cluster new articles",
                )

            # Download button
            if st.button("🚀 Download & Process Articles", type="primary"):
                with st.spinner("🔄 Downloading articles..."):
                    try:
                        # Convert dates to string format
                        from_date_str = (
                            from_date.strftime("%Y-%m-%d") if from_date else None
                        )
                        to_date_str = to_date.strftime("%Y-%m-%d") if to_date else None

                        # Download articles
                        new_articles = get_articles(
                            query=query if query else None,
                            section=section if section else None,
                            from_date=from_date_str,
                            to_date=to_date_str,
                            number_of_articles=number_of_articles,
                        )

                        st.success(f"✅ Downloaded {len(new_articles)} articles!")

                        if auto_process and len(new_articles) > 0:
                            with st.spinner("🔄 Processing articles..."):
                                # Step 1: Text cleaning
                                st.info("🧹 Cleaning text...")
                                cleaned_articles = clean_articles(new_articles)

                                # Step 2: NLP preprocessing
                                st.info("🔤 Processing NLP...")
                                processed_articles = preprocess_articles_nlp(
                                    cleaned_articles
                                )

                                # Step 3: Summarization
                                st.info("📝 Generating summaries...")
                                summarized_articles = summarize_articles(
                                    processed_articles
                                )

                                st.success("✅ Processing complete!")

                                # Display results
                                st.markdown("### 📊 Processing Results")

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        "Articles Processed", len(summarized_articles)
                                    )

                                with col2:
                                    avg_summary_length = summarized_articles[
                                        "summary_length"
                                    ].mean()
                                    st.metric(
                                        "Avg Summary Length",
                                        f"{avg_summary_length:.0f} words",
                                    )

                                with col3:
                                    avg_compression = summarized_articles[
                                        "compression_ratio"
                                    ].mean()
                                    st.metric(
                                        "Avg Compression", f"{avg_compression:.1%}"
                                    )

                                # Show sample articles
                                st.markdown("### 📰 Sample Processed Articles")

                                sample_size = min(3, len(summarized_articles))
                                for i in range(sample_size):
                                    article = summarized_articles.iloc[i]

                                    with st.expander(f"📄 {article['Title'][:60]}..."):
                                        st.markdown(f"**Author:** {article['Author']}")
                                        st.markdown(
                                            f"**Section:** {article['Section']}"
                                        )
                                        st.markdown(
                                            f"**Published:** {article['Published Date']}"
                                        )
                                        st.markdown(
                                            f"**Summary:** {article['summary']}"
                                        )
                                        st.markdown(
                                            f"**URL:** [Read full article]({article['Url']})"
                                        )

                                # Store last processed batch in session state for later database update
                                st.session_state["last_processed_articles"] = (
                                    summarized_articles.copy()
                                )

                                # Option to save processed articles
                                st.markdown("### 💾 Save Processed Articles")

                                csv_data = summarized_articles.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Processed Articles (CSV)",
                                    data=csv_data,
                                    file_name=f"processed_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )

                        else:
                            # Show raw articles
                            st.markdown("### 📰 Downloaded Articles")
                            st.dataframe(
                                new_articles[
                                    ["Title", "Section", "Author", "Published Date"]
                                ]
                            )

                            # Download raw articles
                            csv_data = new_articles.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Raw Articles (CSV)",
                                data=csv_data,
                                file_name=f"raw_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )

                    except Exception as e:
                        st.error(f"❌ Error downloading articles: {str(e)}")
                        st.info(
                            "💡 Make sure your Guardian API key is configured in config.py"
                        )

            # Separate section: add last processed batch to existing database
            if "last_processed_articles" in st.session_state:
                st.markdown("### 🗄️ Database Update")
                st.caption(
                    "You can add the last processed batch of articles to the existing database."
                )

                if st.button("🗄️ Add Last Processed Articles to Existing Database"):
                    summarized_articles = st.session_state["last_processed_articles"]

                    with st.spinner("🔄 Adding to database..."):
                        # Get article count before update
                        before_stats = get_database_stats()
                        before_total = before_stats.get("total_articles", 0)

                        success = update_article_database(
                            summarized_articles,
                            reprocess_topics=True,
                        )

                        if success:
                            # Recompute stats to see how many new articles were added
                            after_stats = get_database_stats()
                            after_total = after_stats.get("total_articles", before_total)
                            added = max(after_total - before_total, 0)

                            if added > 0:
                                st.success(
                                    f"✅ Articles added to database! ({added} new)"
                                )
                            else:
                                st.info(
                                    "ℹ️ No new unique articles were added (all URLs already exist in the database)."
                                )

                            # Force refresh of database stats
                            st.session_state.db_refresh_key = (
                                st.session_state.get("db_refresh_key", 0) + 1
                            )
                            st.info("🔄 Refreshing database stats...")
                            st.rerun()
                        else:
                            st.error("❌ Failed to add articles to database.")

            # Processing pipeline info
            st.markdown("### ℹ️ Processing Pipeline")
            st.info(
                """
            **The processing pipeline includes:**
            1. 🧹 **Text Cleaning** - Remove HTML, normalize text
            2. 🔤 **NLP Processing** - Tokenization, stop words removal, lemmatization
            3. 📝 **Summarization** - TextRank algorithm for concise summaries
            4. 🎯 **Topic Modeling** - TF-IDF + K-Means clustering (if enabled)
            """
            )

            # API configuration info
            st.markdown("### ⚙️ API Configuration")
            st.info(
                """
            **Guardian API Setup:**
            1. Get your free API key from [The Guardian Developer](https://open-platform.theguardian.com/)
            2. Add your API key to `config.py`
            3. Free tier allows 5,000 requests per day
            """
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "### 🚀 Smart News Summarizer & Analyzer Dashboard\n"
        "Built with Streamlit | Powered by TextRank, TF-IDF, and K-Means clustering"
    )


if __name__ == "__main__":
    main()
