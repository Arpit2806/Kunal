import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Banking Association Rule Dashboard",
    layout="wide"
)

st.title("🏦 Retail Banking Association Rule Mining Dashboard")
st.markdown("**Interactive Market Basket Analysis on Clean Banking Dataset**")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = "Clean_data_retail_banking_customers.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

st.success("✅ Dataset Loaded Successfully")

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
with st.expander("📄 Preview Dataset"):
    st.dataframe(df.head())

# --------------------------------------------------
# PREPROCESSING (BINARY ENCODING)
# --------------------------------------------------
st.subheader("⚙️ Data Preparation")

# Keep only binary columns (0/1 or Yes/No)
binary_df = df.copy()

for col in binary_df.columns:
    binary_df[col] = binary_df[col].apply(lambda x: 1 if x == 1 or x == "Yes" else 0)

st.write("Binary Encoded Dataset Shape:", binary_df.shape)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("🎛 Rule Mining Controls")

min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.6, 0.05)
min_lift = st.sidebar.slider("Minimum Lift", 0.5, 5.0, 1.0, 0.1)

# --------------------------------------------------
# FREQUENT ITEMSETS
# --------------------------------------------------
st.subheader("📌 Frequent Itemsets")

frequent_itemsets = apriori(
    binary_df,
    min_support=min_support,
    use_colnames=True
)

st.write("Total Itemsets Found:", len(frequent_itemsets))

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

st.dataframe(frequent_itemsets.head(20), use_container_width=True)

# --------------------------------------------------
# ASSOCIATION RULES
# --------------------------------------------------
st.subheader("🔗 Association Rules")

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=min_confidence
)

rules = rules[rules["lift"] >= min_lift]
rules = rules.sort_values(["confidence", "lift"], ascending=False)

st.write("Total Rules Generated:", len(rules))
st.dataframe(rules.head(20), use_container_width=True)

# --------------------------------------------------
# TOP 10 RULES KPI
# --------------------------------------------------
st.subheader("🏆 Top 10 Associations")

top10 = rules.head(10)[
    ["antecedents", "consequents", "support", "confidence", "lift"]
].copy()

top10["antecedents"] = top10["antecedents"].astype(str)
top10["consequents"] = top10["consequents"].astype(str)

st.dataframe(top10, use_container_width=True)

# --------------------------------------------------
# VISUALIZATION 1 — SUPPORT BAR CHART
# --------------------------------------------------
st.subheader("📊 Top Itemsets by Support")

top_items = frequent_itemsets.head(10)

fig1 = px.bar(
    top_items,
    x="support",
    y=top_items["itemsets"].astype(str),
    orientation="h",
    title="Top 10 Frequent Itemsets"
)

st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------
# VISUALIZATION 2 — CONFIDENCE BAR CHART
# --------------------------------------------------
st.subheader("📈 Top Rules by Confidence")

top_rules = rules.head(10)

fig2 = px.bar(
    top_rules,
    x="confidence",
    y=top_rules["antecedents"].astype(str),
    orientation="h",
    title="Top 10 Rules by Confidence"
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# VISUALIZATION 3 — LIFT VS CONFIDENCE SCATTER
# --------------------------------------------------
st.subheader("📉 Lift vs Confidence")

fig3 = px.scatter(
    rules,
    x="confidence",
    y="lift",
    size="support",
    hover_data=["antecedents", "consequents"],
    title="Lift vs Confidence Scatter Plot"
)

st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# VISUALIZATION 4 — HEATMAP
# --------------------------------------------------
st.subheader("🔥 Metric Correlation Heatmap")

heatmap_data = rules[["support", "confidence", "lift", "leverage", "conviction"]]

fig4, ax = plt.subplots()
sns.heatmap(
    heatmap_data.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)
st.pyplot(fig4)

# --------------------------------------------------
# VISUALIZATION 5 — NETWORK GRAPH
# --------------------------------------------------
st.subheader("🕸 Association Network Map")

G = nx.DiGraph()

network_rules = rules.head(15)

for _, row in network_rules.iterrows():
    a = list(row["antecedents"])[0]
    c = list(row["consequents"])[0]
    G.add_edge(a, c, weight=row["confidence"])

pos = nx.spring_layout(G, seed=42)

fig5, ax = plt.subplots(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2500,
    font_size=10,
    ax=ax
)

st.pyplot(fig5)

# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------
st.subheader("💡 Business Insights")

st.markdown("""
### 🎯 How to Interpret This Dashboard

- **High Support** → Products widely adopted together  
- **High Confidence** → Strong upsell potential  
- **High Lift** → True association beyond chance  
- **Network Map** → Visual cross-sell pathways  

### 📌 Business Use-Cases
✔ Product Bundling  
✔ Personalized Campaigns  
✔ Cross-selling Strategy  
✔ Portfolio Optimization  
✔ Revenue Uplift  

This dashboard converts analytical results into actionable business intelligence.
""")
