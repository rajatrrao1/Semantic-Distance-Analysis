import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GroupSemanticAnalyzer:
    """
    Compute within-group semantic distances
    for arbitrary groups of texts in a CSV.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def compute_embeddings(self, texts):
        """Encode a list of texts into sentence embeddings."""
        clean_texts = ["" if (t is None) else str(t) for t in texts]
        return self.model.encode(clean_texts, show_progress_bar=False)

    def compute_within_distances(self, embeddings):
        """
        For a set of embeddings, compute all pairwise distances
        (1 - cosine similarity), upper triangle without diagonal.
        """
        sims = cosine_similarity(embeddings)
        dists = 1 - sims
        np.fill_diagonal(dists, 0.0)
        upper = dists[np.triu_indices_from(dists, k=1)]
        return upper

    def summarize_group(self, group_name, texts):
        """Return embeddings, distances, and stats for a single group."""
        embeddings = self.compute_embeddings(texts)
        within = self.compute_within_distances(embeddings)

        stats = {
            "group": group_name,
            "n_texts": int(len(texts)),
            "mean_distance": float(np.mean(within)),
            "std_distance": float(np.std(within)),
            "min_distance": float(np.min(within)),
            "max_distance": float(np.max(within)),
            "median_distance": float(np.median(within)),
            "q25_distance": float(np.percentile(within, 25)),
            "q75_distance": float(np.percentile(within, 75)),
        }

        return {
            "name": group_name,
            "texts": texts,
            "embeddings": embeddings,
            "within_distances": within,
            "stats": stats,
        }

    def analyze_groups(self, df, text_col, group_col, selected_groups):
        """
        Compute within-group stats for each selected group.

        Returns:
            results: dict[group_name] -> dict with embeddings, distances, stats
            stats_df: DataFrame with one row per group
        """
        results = {}
        stats_rows = []

        for g in selected_groups:
            subset_series = df[df[group_col] == g][text_col]

            subset = (
                subset_series
                .dropna()
                .astype(str)
                .tolist()
            )

            if len(subset) < 2:
                continue

            res = self.summarize_group(g, subset)
            results[g] = res
            stats_rows.append(res["stats"])

        stats_df = pd.DataFrame(stats_rows)
        if not stats_df.empty:
            stats_df = stats_df.sort_values("mean_distance", ascending=False).reset_index(drop=True)

        return results, stats_df
