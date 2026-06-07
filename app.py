import streamlit as st
import matplotlib.pyplot as plt

from logic import CyberThreatModel

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Cyber Threat Intelligence Center",
    layout="wide"
)

# ==================================================
# STYLING
# ==================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #06111d,
        #0b1d33,
        #102946
    );
}

.main-title{
    font-size:48px;
    font-weight:800;
    text-align:center;
    color:#00d4ff;
    margin-bottom:0px;
}

.subtitle{
    text-align:center;
    color:#8da9c4;
    font-size:16px;
    margin-bottom:25px;
}

div[data-testid="metric-container"]{
    background:#10243b;
    border:1px solid #1d4c74;
    padding:15px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = CyberThreatModel()

    model.preprocess()
    model.run_kmeans()

    return model

model = load_model()

# ==================================================
# HEADER
# ==================================================

st.markdown(
"""
<div class="main-title">
CYBER THREAT INTELLIGENCE CENTER
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">
Network Traffic Analysis • Threat Clustering • Pattern Discovery
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ==================================================
# METRICS
# ==================================================

metrics = model.get_metrics()
summary = model.threat_summary()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Network Events",
        f"{metrics['total_records']:,}"
    )

with c2:
    st.metric(
        "Threat Events",
        f"{metrics['attack_records']:,}"
    )

with c3:
    st.metric(
        "Safe Events",
        f"{metrics['normal_records']:,}"
    )

with c4:
    st.metric(
        "Threat Clusters",
        metrics["clusters"]
    )

with c5:
    st.metric(
        "Primary Threat",
        summary["top_attack"]
    )

st.divider()

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Threat Intelligence",
    "Threat Landscape",
    "Pattern Mining",
    "Threat Explorer"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Cluster Risk Assessment")

    cluster_summary = model.cluster_summary()

    risks = model.cluster_risk_levels()

    for cluster_id, count in cluster_summary.items():

        risk = risks[cluster_id]["label"]

        if risk == "HIGH RISK":

            st.error(
                f"Cluster {cluster_id} | {count:,} events | HIGH RISK"
            )

        elif risk == "MEDIUM RISK":

            st.warning(
                f"Cluster {cluster_id} | {count:,} events | MEDIUM RISK"
            )

        else:

            st.success(
                f"Cluster {cluster_id} | {count:,} events | LOW RISK"
            )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("Threat Landscape Mapping")

    pca_df = model.get_pca_data()

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    colors = {
        0: "green",
        1: "orange",
        2: "red"
    }

    for cluster in pca_df["Cluster"].unique():

        subset = pca_df[
            pca_df["Cluster"] == cluster
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            alpha=0.5,
            label=f"Cluster {cluster}",
            c=colors.get(cluster, "blue")
        )

    ax.legend()

    ax.set_title(
        "Network Traffic Clustering"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    st.pyplot(fig)

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.subheader(
        "Discovered Attack Patterns"
    )

    with st.spinner(
        "Mining Association Rules..."
    ):

        rules = model.generate_rules()

    st.dataframe(
        rules,
        use_container_width=True,
        height=600
    )

# ==================================================
# TAB 4
# ==================================================

with tab4:

    cluster_id = st.selectbox(
        "Select Threat Cluster",
        sorted(
            model.df["cluster"].unique()
        )
    )

    attacks = model.analyze_cluster(
        cluster_id
    )

    st.subheader(
        f"Cluster {cluster_id} Analysis"
    )

    st.bar_chart(attacks)

    st.dataframe(
        attacks.reset_index(),
        use_container_width=True
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Threat Intelligence Engine Active"
)