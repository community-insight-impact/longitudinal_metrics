import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps

import warnings
warnings.filterwarnings("ignore")

severity_historical = pd.read_csv("https://raw.githubusercontent.com/community-insight-impact/longitudinal_metrics/main/severity_historical.csv")

selectcols = severity_historical.columns.tolist()
selectcols.remove("FIPS")
selectcols.remove("year")

st.set_page_config(layout="wide")
st.title("Exploratory Analyses")
st.subheader("View relationships within & between features over time")

d = st.sidebar.selectbox(label = "Select a year:", 
                         options = severity_historical["year"].unique().tolist(),
                         index = 0)

l = st.sidebar.selectbox(label = "Select a variable (for univariate distributions):",
                        options = selectcols,
                        index = 0)

subsetdf = severity_historical.loc[severity_historical["year"] == d].drop(labels = ["FIPS", "year"], axis = 1)

col1, col2, col3 = st.beta_columns(3)

matrix_df = pps.matrix(subsetdf)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

fig1 = plt.figure(1)
sns.histplot(subsetdf, x = l).set_title("Univariate Distribution")

fig2 = plt.figure(2)
sns.heatmap(subsetdf.corr(), cmap="flare", annot=True).set_title("Correlation Matrix")

fig3 = plt.figure(3)
sns.heatmap(matrix_df, cmap="cubehelix_r", annot=True).set_title("Predictive Power Score")

col1.pyplot(fig1, use_container_widths = True)
col2.pyplot(fig2, use_container_widths = True)
col3.pyplot(fig3, use_container_widths = True)