import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import GroupSemanticAnalyzer


# Basic theming for plots
sns.set_style("whitegrid")

st.set_page_config(
    page_title="Semantic Distance Analysis",
    layout="wide"
)

# ---- HEADER ----
st.markdown(
    """
    <style>
    .big-title {
        font-size: 32px;
        font-weight: 700;
        padding-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 14px;
        color: #666666;
        padding-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Semantic Distance Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Upload a CSV, pick a text column and a group column, and explore how semantically '
    'diverse each group is.'
    '</div>',
    unsafe_allow_html=True,
)

# ---- LAYOUT CONTAINERS ----
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    # Two-column layout for controls vs preview
    controls_col, preview_col = st.columns([2, 3])

    with preview_col:
        st.subheader("Data preview")
        st.dataframe(df.head(), use_container_width=True)

    with controls_col:
        st.subheader("Settings")

        model_name = st.selectbox(
            "Embedding model",
            options=[
                "BAAI/bge-large-en-v1.5",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "bert-base-uncased",
            ],
            index=0,
            help=(
                "Choose a model for computing sentence embeddings. "
                "Smaller models (MiniLM) are faster; larger ones (bge-large, mpnet) can capture more nuance."
            )
        )

        text_col = st.selectbox(
            "Text column",
            options=df.columns,
            help="Column containing the texts (descriptions, comments, etc.)."
        )

        group_col = st.selectbox(
            "Group column",
            options=df.columns,
            help="Column defining groups (e.g. artwork_number, condition, dataset name)."
        )

        unique_groups = df[group_col].dropna().unique().tolist()
        if len(unique_groups) == 0:
            st.error("No groups found in the selected group column.")
            st.stop()

        # All groups selected by default
        default_groups = unique_groups

        selected_groups = st.multiselect(
            "Groups included in the analysis",
            options=unique_groups,
            default=default_groups,
            help="All groups are selected by default. Remove any you don't want."
        )

        min_texts = st.slider(
            "Minimum texts per group",
            min_value=2,
            max_value=20,
            value=2,
            help="Groups with fewer texts than this will be ignored in the analysis."
        )

        run_button = st.button("Run semantic distance analysis", type="primary")

    if run_button:
        with st.spinner("Computing embeddings and within-group distances..."):
            # Filter out groups that are too small before analysis
            valid_groups = []
            for g in selected_groups:
                n = df[df[group_col] == g][text_col].dropna().shape[0]
                if n >= min_texts:
                    valid_groups.append(g)

            if len(valid_groups) == 0:
                st.error("No group has enough texts given the minimum threshold.")
                st.stop()

            analyzer = GroupSemanticAnalyzer(model_name=model_name)
            results, stats_df = analyzer.analyze_groups(
                df=df,
                text_col=text_col,
                group_col=group_col,
                selected_groups=valid_groups
            )

        if stats_df.empty:
            st.error("No group had at least 2 valid texts after cleaning.")
            st.stop()

        st.success("Analysis complete.")

        # ---- MAIN RESULTS LAYOUT ----
        top_col, table_col = st.columns([2, 3])

        with table_col:
            st.subheader("Within-group semantic distance statistics")
            st.dataframe(stats_df, use_container_width=True)

        with top_col:
            st.subheader("Summary")
            st.markdown(
                f"- Number of groups analysed: **{stats_df.shape[0]}**\n"
                f"- Total texts analysed: **{int(stats_df['n_texts'].sum())}**\n"
                f"- Highest mean distance: **{stats_df.iloc[0]['group']}** "
                f"({stats_df.iloc[0]['mean_distance']:.3f})\n"
                f"- Lowest mean distance: **{stats_df.iloc[-1]['group']}** "
                f"({stats_df.iloc[-1]['mean_distance']:.3f})"
            )

        st.markdown("---")

        # ---- VISUALS ----
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.subheader("Mean distances by group")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=stats_df,
                x="group",
                y="mean_distance",
                ax=ax1,
                palette="viridis"
            )
            ax1.set_xlabel("Group")
            ax1.set_ylabel("Mean semantic distance")
            ax1.set_title("Mean within-group distances")
            ax1.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)

        with plot_col2:
            st.subheader("Distance variability by group")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=stats_df,
                x="n_texts",
                y="mean_distance",
                size="std_distance",
                hue="std_distance",
                palette="magma",
                ax=ax2,
                legend=False
            )
            ax2.set_xlabel("Number of texts")
            ax2.set_ylabel("Mean distance")
            ax2.set_title("Sample size vs semantic diversity")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        st.subheader("Within-group distance distributions (up to 8 groups)")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        dist_data = []
        labels = []
        for g in stats_df["group"].tolist()[:8]:
            dist_data.append(results[g]["within_distances"])
            labels.append(str(g))

        if dist_data:
            ax3.boxplot(dist_data, labels=labels)
            ax3.set_xlabel("Group")
            ax3.set_ylabel("Semantic distance")
            ax3.set_title("Within-group distance distributions")
            ax3.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
        else:
            st.info("Not enough texts per group for distance distributions.")

        # ---- DOWNLOAD ----
        st.subheader("Download results")
        stats_csv = stats_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download within-group stats (CSV)",
            data=stats_csv,
            file_name="semantic_distance_within_group_stats.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a CSV file to begin.")
