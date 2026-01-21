import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Banking Analytics Dashboard",
    layout="wide"
)

st.title("🏦 Banking Analytics Dashboard")
st.markdown("**Association Rules + K-Means Clustering on Retail Banking Data**")

# ======================================================
# LOAD DATA (COMMON)
# ======================================================
DATA_PATH = "Clean_data_retail_banking_customers.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Select Analysis Page",
    ["📊 Association Rule Mining", "📈 K-Means Clustering"]
)

# ======================================================
# ================= ARM PAGE ===========================
# ======================================================
if page == "📊 Association Rule Mining":

    st.header("📊 Association Rule Mining")

    with st.expander("📄 Dataset Preview"):
        st.dataframe(df.head())

    # -----------------------------
    # Binary Encoding
    # -----------------------------
    binary_df = df.copy()
    for col in binary_df.columns:
        binary_df[col] = binary_df[col].apply(lambda x: 1 if x == 1 or x == "Yes" else 0)

    # -----------------------------
    # Sidebar Controls
    # -----------------------------
    st.sidebar.subheader("⚙️ ARM Controls")
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.6, 0.05)
    min_lift = st.sidebar.slider("Minimum Lift", 0.5, 5.0, 1.0, 0.1)

    # -----------------------------
    # Frequent Itemsets
    # -----------------------------
    st.subheader("📌 Frequent Itemsets")

    frequent_itemsets = apriori(
        binary_df,
        min_support=min_support,
        use_colnames=True
    )

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

    st.write("Total Itemsets:", len(frequent_itemsets))
    st.dataframe(frequent_itemsets.head(20), use_container_width=True)

    # -----------------------------
    # Association Rules
    # -----------------------------
    st.subheader("🔗 Association Rules")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules = rules[rules["lift"] >= min_lift]
    rules = rules.sort_values(["confidence", "lift"], ascending=False)

    # Fix frozenset → string
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    st.write("Total Rules:", len(rules))
    st.dataframe(rules.head(20), use_container_width=True)

    # -----------------------------
    # Top 10 Rules
    # -----------------------------
    st.subheader("🏆 Top 10 Associations")

    top10 = rules.head(10)[
        ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    ]
    st.dataframe(top10, use_container_width=True)

    # -----------------------------
    # Visualizations
    # -----------------------------
    st.subheader("📊 Visual Analytics")

    col1, col2 = st.columns(2)

    with col1:
        top_items = frequent_itemsets.head(10).copy()
        top_items["itemsets_str"] = top_items["itemsets"].astype(str)

        fig1 = px.bar(
            top_items,
            x="support",
            y="itemsets_str",
            orientation="h",
            title="Top Itemsets by Support"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        top_rules = rules.head(10).copy()

        fig2 = px.bar(
            top_rules,
            x="confidence",
            y="antecedents_str",
            orientation="h",
            title="Top Rules by Confidence"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter
    fig3 = px.scatter(
        rules,
        x="confidence",
        y="lift",
        size="support",
        hover_data=["antecedents_str", "consequents_str"],
        title="Lift vs Confidence"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap
    st.subheader("🔥 Metric Correlation Heatmap")

    heatmap_data = rules[["support", "confidence", "lift", "leverage", "conviction"]]
    fig4, ax = plt.subplots()
    sns.heatmap(heatmap_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig4)

    # Network Graph
    st.subheader("🕸 Association Network")

    G = nx.DiGraph()
    network_rules = rules.head(15)

    for _, row in network_rules.iterrows():
        G.add_edge(row["antecedents_str"], row["consequents_str"])

    pos = nx.spring_layout(G, seed=42)

    fig5, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2500, font_size=10, ax=ax)
    st.pyplot(fig5)

# ======================================================
# ================= K-MEANS PAGE ========================
# ======================================================
elif page == "📈 K-Means Clustering":

    st.header("📈 K-Means Clustering")

    with st.expander("📄 Dataset Preview"):
        st.dataframe(df.head())

    # -----------------------------
    # Numeric Columns
    # -----------------------------
    num_df = df.select_dtypes(include=np.number)

    st.write("Numeric Features Used:", list(num_df.columns))

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(num_df)

    # -----------------------------
    # Sidebar Controls
    # -----------------------------
    st.sidebar.subheader("⚙️ Clustering Controls")
    k = st.sidebar.slider("Number of Clusters (K)", 2, 8, 3)

    # -----------------------------
    # Elbow Method
    # -----------------------------
    st.subheader("📉 Elbow Method")

    wcss = []
    K_range = range(2, 9)
    for i in K_range:
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(scaled_data)
        wcss.append(km.inertia_)

    fig6 = px.line(
        x=list(K_range),
        y=wcss,
        markers=True,
        title="Elbow Curve (WCSS vs K)",
        labels={"x": "Number of Clusters", "y": "WCSS"}
    )
    st.plotly_chart(fig6, use_container_width=True)

    # -----------------------------
    # Silhouette Score
    # -----------------------------
    st.subheader("📐 Silhouette Score")

    sil_scores = []
    for i in K_range:
        labels = KMeans(n_clusters=i, random_state=42).fit_predict(scaled_data)
        sil_scores.append(silhouette_score(scaled_data, labels))

    fig7 = px.line(
        x=list(K_range),
        y=sil_scores,
        markers=True,
        title="Silhouette Score vs K",
        labels={"x": "Number of Clusters", "y": "Silhouette Score"}
    )
    st.plotly_chart(fig7, use_container_width=True)

    # -----------------------------
    # KMeans Model
    # -----------------------------
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    df_clustered = num_df.copy()
    df_clustered["Cluster"] = clusters

    st.subheader("📊 Cluster Distribution")
    st.dataframe(df_clustered["Cluster"].value_counts().reset_index())

    # -----------------------------
    # PCA Visualization
    # -----------------------------
    st.subheader("📍 PCA Cluster Visualization")

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters.astype(str)

    fig8 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="Customer Segmentation using PCA + KMeans"
    )
    st.plotly_chart(fig8, use_container_width=True)

    # -----------------------------
    # Cluster Profiling
    # -----------------------------
    st.subheader("🧮 Cluster Profiling")

    cluster_profile = df_clustered.groupby("Cluster").mean().round(2)
    st.dataframe(cluster_profile, use_container_width=True)

    # -----------------------------
    # Business Interpretation
    # -----------------------------
    st.subheader("💡 Business Insights")

    st.markdown("""
    - Clusters represent distinct customer segments based on behavioral attributes.
    - PCA visualization helps understand cluster separation.
    - Elbow + Silhouette support optimal cluster selection.
    - Cluster profiling supports targeted marketing and personalization.
    - Enables segmentation-driven cross-selling strategy alignment.
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("🚀 Built for MBA Analytics Portfolio | Streamlit Dashboard")
