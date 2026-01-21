import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="ARM Business Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Association Rule Mining Dashboard")
st.markdown("**Interactive Business Insights from Market Basket Model**")

# -------------------------------
# LOAD MODEL
# -------------------------------
MODEL_PATH = "ARM model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# UNDERSTAND MODEL CONTENT
# -------------------------------
st.subheader("📦 Model Contents")
st.write(type(model))

if isinstance(model, dict):
    st.write("Keys inside model:", model.keys())

# Expecting keys like:
# model["frequent_itemsets"]
# model["rules"]

frequent_itemsets = None
rules = None

if isinstance(model, dict):
    frequent_itemsets = model.get("frequent_itemsets")
    rules = model.get("rules")

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🎛 Filters")

min_support = st.sidebar.slider(
    "Minimum Support",
    0.01, 1.0, 0.05, 0.01
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.1, 1.0, 0.5, 0.05
)

min_lift = st.sidebar.slider(
    "Minimum Lift",
    0.5, 5.0, 1.0, 0.1
)

# -------------------------------
# FREQUENT ITEMSETS
# -------------------------------
if frequent_itemsets is not None:
    st.subheader("📌 Frequent Itemsets")

    fi_df = frequent_itemsets.copy()
    fi_df = fi_df[fi_df["support"] >= min_support]
    fi_df = fi_df.sort_values("support", ascending=False)

    st.dataframe(fi_df, use_container_width=True)

    # ---- Visualization
    st.subheader("📈 Top Itemsets by Support")

    top_items = fi_df.head(10)

    fig, ax = plt.subplots()
    ax.barh(
        top_items["itemsets"].astype(str),
        top_items["support"]
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Itemsets")
    ax.invert_yaxis()

    st.pyplot(fig)

else:
    st.warning("⚠️ Frequent itemsets not found in model.")

# -------------------------------
# ASSOCIATION RULES
# -------------------------------
if rules is not None:
    st.subheader("🔗 Association Rules")

    rules_df = rules.copy()
    rules_df = rules_df[
        (rules_df["confidence"] >= min_confidence) &
        (rules_df["lift"] >= min_lift)
    ].sort_values("confidence", ascending=False)

    st.dataframe(rules_df, use_container_width=True)

    # ---- Visualization 1: Confidence
    st.subheader("📊 Top Rules by Confidence")

    top_rules = rules_df.head(10)

    fig2, ax2 = plt.subplots()
    ax2.barh(
        top_rules["antecedents"].astype(str),
        top_rules["confidence"]
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Antecedents")
    ax2.invert_yaxis()

    st.pyplot(fig2)

    # ---- Visualization 2: Lift vs Confidence
    st.subheader("📉 Lift vs Confidence")

    fig3, ax3 = plt.subplots()
    ax3.scatter(
        rules_df["confidence"],
        rules_df["lift"]
    )
    ax3.set_xlabel("Confidence")
    ax3.set_ylabel("Lift")

    st.pyplot(fig3)

else:
    st.warning("⚠️ Association rules not found in model.")

# -------------------------------
# BUSINESS INSIGHTS
# -------------------------------
st.subheader("💡 Business Interpretation")

st.markdown("""
- **High Support Itemsets** → Popular product combinations  
- **High Confidence Rules** → Strong cross-sell opportunity  
- **High Lift Rules** → True business impact beyond random chance  

👉 These insights help in:
- Product bundling  
- Targeted marketing  
- Cross-selling strategy  
- Customer segmentation  
""")
