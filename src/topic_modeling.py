"""
Topic modeling using TF-IDF, PCA, and K-Means clustering.

This module provides functions to perform topic modeling on articles,
including TF-IDF vectorization, dimensionality reduction with PCA,
and clustering with K-Means.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils_nlp import tokenize_text

# Comprehensive stop words list for better filtering
STOP_WORDS = {
    # Basic articles and pronouns
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
    "get",
    "got",
    "go",
    "went",
    "come",
    "came",
    "see",
    "saw",
    "know",
    "knew",
    "think",
    "thought",
    "say",
    "said",
    "tell",
    "told",
    "make",
    "made",
    "take",
    "took",
    "give",
    "gave",
    "find",
    "found",
    "look",
    "looked",
    "use",
    "used",
    "work",
    "worked",
    "call",
    "called",
    "try",
    "tried",
    "ask",
    "asked",
    "need",
    "needed",
    "feel",
    "felt",
    "become",
    "became",
    "leave",
    "left",
    "put",
    "keep",
    "kept",
    "let",
    "begin",
    "began",
    "seem",
    "seemed",
    "help",
    "helped",
    "show",
    "showed",
    "hear",
    "heard",
    "play",
    "played",
    "run",
    "ran",
    "move",
    "moved",
    "live",
    "lived",
    "believe",
    "believed",
    "hold",
    "held",
    "bring",
    "brought",
    "happen",
    "happened",
    "write",
    "wrote",
    "sit",
    "sat",
    "stand",
    "stood",
    "lose",
    "lost",
    "pay",
    "paid",
    "meet",
    "met",
    "include",
    "included",
    "continue",
    "continued",
    "set",
    "turn",
    "turned",
    "start",
    "started",
    "create",
    "created",
    "provide",
    "provided",
    "follow",
    "followed",
    "stop",
    "stopped",
    "produce",
    "produced",
    "build",
    "built",
    "allow",
    "allowed",
    "add",
    "added",
    "spend",
    "spent",
    "grow",
    "grew",
    "open",
    "opened",
    "walk",
    "walked",
    "win",
    "won",
    "offer",
    "offered",
    "remember",
    "remembered",
    "love",
    "loved",
    "consider",
    "considered",
    "appear",
    "appeared",
    "buy",
    "bought",
    "wait",
    "waited",
    "serve",
    "served",
    "die",
    "died",
    "send",
    "sent",
    "expect",
    "expected",
    "build",
    "built",
    "stay",
    "stayed",
    "fall",
    "fell",
    "cut",
    "reach",
    "reached",
    "kill",
    "killed",
    "remain",
    "remained",
    "suggest",
    "suggested",
    "raise",
    "raised",
    "pass",
    "passed",
    "sell",
    "sold",
    "require",
    "required",
    "report",
    "reported",
    "decide",
    "decided",
    "pull",
    "pulled",
    # Additional common words that appear in news
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "first",
    "second",
    "third",
    "last",
    "next",
    "previous",
    "new",
    "old",
    "good",
    "bad",
    "big",
    "small",
    "long",
    "short",
    "high",
    "low",
    "great",
    "little",
    "much",
    "many",
    "few",
    "several",
    "every",
    "each",
    "both",
    "him",
    "her",
    "his",
    "hers",
    "himself",
    "herself",
    "itself",
    "themselves",
    "us",
    "our",
    "ours",
    "ourselves",
    "yourself",
    "yourselves",
    "bst",
    "pm",
    "am",
    "gmt",
    "utc",
    "time",
    "day",
    "night",
    "morning",
    "evening",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "year",
    "month",
    "week",
    "hour",
    "minute",
    "second",
    "away",
    "goes",
    "went",
    "gone",
    "coming",
    "coming",
    "came",
    "confirmed",
    "according",
    "accordingly",
    "however",
    "therefore",
    "thus",
    "meanwhile",
    "furthermore",
    "moreover",
    "nevertheless",
    "nonetheless",
    "people",
    "person",
    "man",
    "woman",
    "child",
    "children",
    "family",
    "government",
    "official",
    "minister",
    "president",
    "prime",
    "minister",
    "police",
    "officer",
    "soldier",
    "military",
    "army",
    "navy",
    "air",
    "company",
    "business",
    "industry",
    "economy",
    "economic",
    "financial",
    "public",
    "private",
    "national",
    "international",
    "local",
    "global",
    "social",
    "political",
    "cultural",
    "environmental",
    "educational",
    "health",
    "medical",
    "scientific",
    "technological",
    "digital",
    "online",
    "internet",
    "website",
    "email",
    "phone",
    "mobile",
    "car",
    "vehicle",
    "transport",
    "travel",
    "flight",
    "train",
    "bus",
    "home",
    "house",
    "building",
    "street",
    "city",
    "town",
    "country",
    "world",
    "earth",
    "planet",
    "space",
    "universe",
    "nature",
    "environment",
}


def create_count_vector(words: list[str]) -> list[int]:
    """
    Create a count vector (term frequency) from a list of words.

    Args:
        words (list[str]): List of words (tokens) from a document.

    Returns:
        list[int]: List of word counts (each unique word's count in the document).
    """
    words = [word.lower() for word in words]
    d = {}
    for word in words:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
    return list(d.values())


def create_idf_vector(texts: list[str], vocabulary) -> list[float]:
    """
    Create an IDF vector aligned to the provided vocabulary order.

    Args:
        texts (list[str]): List of documents (strings).
        vocabulary: Ordered iterable of unique words (e.g., list).

    Returns:
        list[float]: IDF scores in the same order as `vocabulary`.
    """
    if not isinstance(vocabulary, list):
        vocabulary = list(vocabulary)
    num_docs = len(texts)
    if num_docs == 0:
        # Match test expectation: zeros when no documents
        if not isinstance(vocabulary, list):
            vocabulary = list(vocabulary)
        return [0.0] * len(vocabulary)
    doc_token_sets = []
    for text in texts:
        tokens = tokenize_text(text)
        tokens_lower = {t.lower() for t in tokens}
        doc_token_sets.append(tokens_lower)
    idf_scores = []
    for word in vocabulary:
        df = sum(1 for token_set in doc_token_sets if word in token_set)
        idf = np.log((1 + num_docs) / (1 + df)) + 1.0
        idf_scores.append(float(idf))
    return idf_scores


def create_tfidf_matrix(texts: list[str]) -> tuple[np.array, list[str]]:
    """
    Create TF-IDF matrix using sklearn's TfidfVectorizer.

    Args:
        texts (list[str]): List of documents (strings).

    Returns:
        tuple: (tfidf_matrix, feature_names)
            - tfidf_matrix: Dense TF-IDF matrix (rows = documents, columns = terms).
            - feature_names: List of feature names in column order.
    """
    if not texts:
        return np.zeros((0, 0), dtype=float), []

    # Match our tokenization (include single-char tokens)
    vectorizer = TfidfVectorizer(
        token_pattern=r"\b\w+\b",
        lowercase=True,
        norm="l2",
        use_idf=True,
        sublinear_tf=True,
        ngram_range=(1, 1),
        stop_words=list(STOP_WORDS),  # Use our comprehensive stop words
        min_df=3,  # Ignore terms that appear in less than 3 documents (was 2)
        max_df=0.7,  # Ignore terms that appear in more than 70% of documents (was 0.8)
        max_features=1000,  # Limit to top 1000 features to focus on most important words
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()
    return X.toarray(), feature_names


def cluster_articles(
    tfidf_matrix: np.array, clusters_count: int = 3, random_state: int = 42
):
    """
    Cluster articles using KMeans based on their TF-IDF matrix.

    Args:
        tfidf_matrix (np.array): TF-IDF feature matrix where rows correspond to articles and columns to features/words.
        clusters_count (int, optional): Number of clusters for KMeans. Defaults to 3.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (labels, km)
            - labels (np.ndarray): Array of cluster labels assigned to each article.
            - km (KMeans): Fitted KMeans clustering model.
    """
    # Dimensionality reduction to stabilize clustering
    X = tfidf_matrix
    n_docs, n_feats = (
        X.shape if hasattr(X, "shape") else (len(X), len(X[0]) if X else 0)
    )
    # Choose components conservatively, minimum 2
    if n_docs > 1 and n_feats > 1:
        n_components = max(2, min(10, n_docs - 1, n_feats - 1))
        try:
            X = PCA(n_components=n_components, random_state=random_state).fit_transform(
                X
            )
        except Exception:
            # Fallback: use original space if PCA fails
            X = tfidf_matrix

    km = KMeans(
        n_clusters=clusters_count,
        n_init=50,
        max_iter=1000,
        random_state=random_state,
        algorithm="elkan",
    )
    labels = km.fit_predict(X)

    # Compute cluster centers in ORIGINAL TF-IDF space for interpretability/tests
    tfidf = np.asarray(tfidf_matrix, dtype=float)
    n_features = tfidf.shape[1] if tfidf.ndim == 2 else 0
    centers = np.zeros((clusters_count, n_features), dtype=float)
    for c in range(clusters_count):
        mask = labels == c
        if np.any(mask):
            centers[c] = tfidf[mask].mean(axis=0)
        else:
            # Empty cluster: fallback to global mean
            centers[c] = tfidf.mean(axis=0) if tfidf.size else 0.0

    class _KMProxy:
        pass

    km_proxy = _KMProxy()
    km_proxy.cluster_centers_ = centers

    return labels, km_proxy


def get_top_keywords_per_cluster(
    cluster_centers: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> list[list[str]]:
    """
    Extract top keywords for each cluster based on cluster center weights.

    Args:
        cluster_centers: Array of shape (n_clusters, n_features) with cluster centers.
        feature_names: List of feature names in the same order as cluster_centers columns.
        top_n: Number of top keywords to return per cluster.

    Returns:
        List of lists, where each inner list contains top keywords for one cluster.
    """
    if cluster_centers.size == 0 or not feature_names:
        return []

    n_clusters = cluster_centers.shape[0]
    top_keywords = []

    for c in range(n_clusters):
        center = cluster_centers[c]
        # Get indices of top weights (descending order)
        top_indices = np.argsort(center)[::-1][:top_n]
        # Extract keywords, filtering out very short, numeric, or common tokens
        keywords = []
        # Additional common words to filter out from keywords
        common_words = {
            # Reporting / discourse verbs
            "said",
            "says",
            "told",
            "tells",
            "asked",
            "asks",
            "reported",
            "reports",
            "confirmed",
            "confirms",
            # Connectors / adverbs
            "according",
            "accordingly",
            "however",
            "therefore",
            "meanwhile",
            "furthermore",
            "moreover",
            "nevertheless",
            "nonetheless",
            # Generic time words
            "today",
            "yesterday",
            "tomorrow",
            "week",
            "month",
            "year",
            "time",
            # Very generic nouns
            "way",
            "thing",
            "things",
            "part",
            "parts",
            "place",
            "places",
            "case",
            "point",
            "points",
            "fact",
            "facts",
            "issue",
            "issues",
            "problem",
            "problems",
            "question",
            "questions",
            "answer",
            "answers",
            "result",
            "results",
            "change",
            "changes",
            "development",
            "developments",
            "situation",
            "situations",
            "process",
            "processes",
            "system",
            "systems",
            "program",
            "programs",
            "project",
            "projects",
            "plan",
            "plans",
            "policy",
            "policies",
            "decision",
            "decisions",
            "choice",
            "choices",
            "option",
            "options",
            "opportunity",
            "opportunities",
            "challenge",
            "challenges",
            "risk",
            "risks",
            "benefit",
            "benefits",
            "advantage",
            "advantages",
            "disadvantage",
            "disadvantages",
            "success",
            "successes",
            "failure",
            "failures",
            "victory",
            "victories",
            "defeat",
            "defeats",
            "win",
            "wins",
            "loss",
            "losses",
            "gain",
            "gains",
            # Pronouns / articles / auxiliaries that should never be keywords
            "he",
            "she",
            "him",
            "his",
            "her",
            "they",
            "them",
            "their",
            "who",
            "whom",
            "whose",
            "which",
            "the",
            "this",
            "that",
            "these",
            "those",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }

        for idx in top_indices:
            if idx < len(feature_names):
                word = feature_names[idx]
                # Filter: keep words with length >= 3, not pure digits, not common words
                if (
                    len(word) >= 3
                    and not word.isdigit()
                    and word not in common_words
                    and not word.endswith("ing")
                    or len(word) > 4
                ):  # Avoid very short -ing words
                    keywords.append(word)
        top_keywords.append(keywords)

    return top_keywords


def assign_topics_to_articles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Assign topic labels to articles DataFrame.

    Args:
        df: DataFrame with articles (must have same length as labels).
        labels: Array of cluster labels for each article.

    Returns:
        DataFrame with added 'Topic' column containing cluster labels.
    """
    if len(df) != len(labels):
        raise ValueError(
            f"DataFrame length ({len(df)}) must match labels length ({len(labels)})"
        )

    result_df = df.copy()
    result_df["Topic"] = labels

    return result_df
