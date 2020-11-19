import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps

import warnings
warnings.filterwarnings("ignore")

severity_historical = pd.read_csv("/Users/admin/Documents/cvi/severity-data/severity_historical.csv")

st.title("Exploratory Analyses")
st.subheader("View relationships within & between features over time")

d = st.sidebar.selectbox(label = "Select a year:", 
                         options = severity_historical["year"].unique().tolist(),
                         index = 0)

l = st.sidebar.selectbox(label = "Select a variable (for univariate distributions):",
                        options = severity_historical.columns.tolist(),
                        index = 0)

subsetdf = severity_historical.loc[severity_historical["year"] == d].drop(labels = ["FIPS", "year"], axis = 1)


col1, col2 = st.beta_columns(2)

matrix_df = pps.matrix(subsetdf)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

fig1 = plt.figure(1)
sns.heatmap(subsetdf.corr(), cmap="flare", annot=True).set_title("Correlation Matrix for {f}".format(f = d))

fig2 = plt.figure(2)
sns.heatmap(matrix_df, cmap="cubehelix_r", annot=True).set_title(("Predictive Power Score for {f}".format(f = d)))

col1.pyplot(fig1, use_container_widths = True)
col2.pyplot(fig2, use_container_widths = True)

#st.pyplot(fig, use_container_width=True)