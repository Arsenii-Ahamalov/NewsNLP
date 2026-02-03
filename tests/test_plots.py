"""
Basic tests for visualization helpers in src/plots.py.

These are smoke tests: they check that plotting functions run on a small
synthetic dataset and produce output files, but they do not validate
visual appearance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plots import save_all_plots


def _make_small_topic_df() -> pd.DataFrame:
    """Create a tiny DataFrame with minimal columns for plotting."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Title": ["A", "B", "C", "D"],
            "Section": ["Sport", "Sport", "News", "News"],
            "Topic": [0, 0, 1, 1],
        }
    )


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_save_all_plots_creates_expected_files(tmp_path):
    """save_all_plots should create plot image files in the target directory."""
    df = _make_small_topic_df()

    # Two topics, small TF-IDF space
    top_keywords = [["goal", "match", "team"], ["election", "vote", "policy"]]
    tfidf_matrix = np.random.rand(len(df), 5)
    labels = np.array(df["Topic"].tolist())
    feature_names = [f"w{i}" for i in range(tfidf_matrix.shape[1])]

    topic_to_name = {0: "Sport", 1: "News"}

    output_dir = tmp_path / "plots"
    result = save_all_plots(
        df,
        top_keywords,
        tfidf_matrix,
        labels,
        feature_names,
        output_dir=str(output_dir),
        topic_to_name=topic_to_name,
    )

    # At least the core plots must be present
    for key in [
        "topic_distribution",
        "top_keywords",
        "topic_scatter",
        "keywords_heatmap",
        "summary",
    ]:
        assert key in result
        path = Path(result[key])
        assert path.exists(), f"{key} file was not created"
