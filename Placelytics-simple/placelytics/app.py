"""Tiny dashboard (SYNTHETIC data).  Run: python -m streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Placelytics", page_icon="📊", layout="wide")
st.title("📊 Placelytics - Campus Placement Dashboard")
st.warning("All data is synthetic (a fictional college), not real placement statistics.")

df = pd.read_csv("data/placements.csv").drop_duplicates()
df["cgpa"] = df.cgpa.fillna(df.groupby("branch").cgpa.transform("median"))

years = st.sidebar.multiselect("Batch year", sorted(df.batch_year.unique()), default=sorted(df.batch_year.unique()))
branches = st.sidebar.multiselect("Branch", sorted(df.branch.unique()), default=sorted(df.branch.unique()))
d = df[df.batch_year.isin(years) & df.branch.isin(branches)]
if d.empty:
    st.error("No data for these filters.")
    st.stop()
p = d[d.placed == 1]

c = st.columns(4)
c[0].metric("Students", len(d))
c[1].metric("Placement rate", f"{100 * d.placed.mean():.1f}%")
c[2].metric("Median package", f"{p.package_lpa.median():.1f} LPA")
c[3].metric("Highest package", f"{p.package_lpa.max():.1f} LPA")

a, b = st.columns(2)
g = (d.groupby("branch").placed.mean() * 100).reset_index()
a.plotly_chart(px.bar(g, x="branch", y="placed", title="Placement % by branch"), width="stretch")
b.plotly_chart(px.histogram(p, x="package_lpa", nbins=20, title="Package distribution (LPA)"), width="stretch")
a2, b2 = st.columns(2)
a2.plotly_chart(px.box(d, x=d.placed.map({0: "Not placed", 1: "Placed"}), y="cgpa", title="CGPA by outcome"), width="stretch")
b2.plotly_chart(px.bar(p.company.value_counts().reset_index(), x="count", y="company", orientation="h", title="Offers by company"), width="stretch")
st.download_button("Download filtered data (CSV)", d.to_csv(index=False).encode(), "placements_filtered.csv")
