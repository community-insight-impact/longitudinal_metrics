import pandas as pd
import os
import time
import json
from urllib.request import urlopen
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset

pd.set_option('display.max_columns', 100)
os.chdir("/Users/alliesaizan/Dropbox/cvi/severity-data/")

# Define helper functions and load main dataset
def edit_fips(x):
    return "0" + x if len(x) < 5 else x

severity_historical = pd.read_csv("severity_historical.csv")
severity_historical["FIPS"] = severity_historical["FIPS"].astype(str).apply(edit_fips)

# Time-series data preprocessing (for K-means clustering)
severity_historical_narrow = severity_historical.loc[severity_historical["year"].isin(["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]), ["FIPS", "year", "severe_cases"]].sort_values(by = "FIPS")

fips = severity_historical_narrow.FIPS.unique()
temp = []

for cfip in fips:
    temp.append(severity_historical_narrow.loc[severity_historical_narrow.FIPS == cfip, "severe_cases"].tolist())
X = to_time_series_dataset(temp)
del cfip, temp

# Identifying the correct number of clusters
"""
for i in range(2, 11):

    t = time.time()
    print(f"Cluster N: {i}")

    km_bis = TimeSeriesKMeans(n_clusters=i, metric="softdtw")
    labels_bis = km_bis.fit_predict(X)

    print(f"\tLabels Calculated, Elapsed: {time.time() - t}")

    sc = silhouette_score(X, labels_bis, metric="softdtw")

    print(f"\tN: {i}, Score: {sc}, Elapsed: {time.time() - t} \n")
"""

# Setting the algorithm as the cluster with the highest silhouette score (from the above loop, it was 3)
km_bis = TimeSeriesKMeans(n_clusters=3, metric="softdtw")
labels_bis = km_bis.fit_predict(X)

# Plot the clusters
centroids = km_bis.cluster_centers_
color_palette = {"0": "#01AFB8", "1": "#196E9F", "2": "#D3D3D3"}
for i in range(0,3):
    col = color_palette[str(i)]
    plt.scatter(X[labels_bis == i , 0] , X[labels_bis == i , 1] , label = i, color = col)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black')
plt.legend()
plt.annotate("The black dots indicate the cluster's centroid", xy = (0,0))
plt.show()

# Append the clusters to the severity dataframe
to_merge = pd.DataFrame({"FIPS":fips, "label":labels_bis})
severity_historical = pd.merge(severity_historical, to_merge, on= "FIPS", how = "left")
severity_historical.head()

# Plot clusters on map
to_merge["label"] = to_merge["label"].astype(str)
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(to_merge, geojson=counties, locations='FIPS', color='label',
                           scope="usa",
                           color_discrete_sequence = ["#01AFB8", "#196E9F", "#D3D3D3"],
                           labels={'label':'Assigned Cluster'} )
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.01
))
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()



# Comparing cluster characteristics
col_dict = {'percent_65plus':"% Aged 65+", 'percent_smokers': "% Smokers",      'percent_diabetes': "% With Diabetes",
       'percent_obese': "% Obese", 'copd_death_rate': "COPD Death Rate", 'hypertension_death_rate': "Hypertension Death Rate",
       'total_heart_disease_death_rate': "Heart Disease Death Rate"}

writer = pd.ExcelWriter("clustering_results.xlsx", engine="xlsxwriter")

for col in col_dict.keys():
    title = col_dict[col]
    groupedDF = severity_historical[["label", col]].groupby("label").describe()
    groupedDF.to_excel(writer, sheet_name = title)

writer.save()

# Data preprocessing (scaling to minimize variance among observations)
#from sklearn.preprocessing import MinMaxScaler
#scaled = MinMaxScaler().fit_transform(X = severity_historical_narrow[["severe_cases"]])
#severity_historical_narrow["severe_cases_scaled"] = scaled


