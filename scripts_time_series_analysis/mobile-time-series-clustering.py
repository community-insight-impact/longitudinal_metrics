import pandas as pd
import numpy as np
import os
import time
import json
from urllib.request import urlopen
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset


pd.set_option('display.max_columns', 100)
os.chdir("/Users/alliesaizan/Dropbox/cvi/mobile-health-data/")

mh_historical = pd.read_csv("output/mobile_health_historical.csv")
mh_historical["FIPS"] = mh_historical["FIPS"].astype(str).apply(lambda x: "0" + x if len(x) < 5 else x)

# Time-series data preprocessing (for K-means clustering)
for col in mh_historical.columns:
    print(mh_historical.loc[mh_historical[col].isnull() == False, "year"].unique())

mh_historical_narrow = mh_historical.loc[mh_historical["year"].isin(["2011", "2012", "2013", "2014", "2015", "2016", "2017"]), ["FIPS", "year", "score"]].sort_values(by = "FIPS")

fips = mh_historical_narrow.FIPS.unique()
temp = []

for cfip in fips:
    temp.append(mh_historical_narrow.loc[mh_historical_narrow.FIPS == cfip, "score"].tolist())
X = to_time_series_dataset(temp)
del cfip, temp

# Identifying the correct number of clusters (using elbow curve)
""" distortions = []
for i in range(3, 13):

    t = time.time()
    print(f"Cluster N: {i}")

    km_bis = TimeSeriesKMeans(n_clusters=i, metric="softdtw")
    labels_bis = km_bis.fit(X)
    distortions.append(labels_bis.inertia_)

    print(f"\tLabels Calculated, Elapsed: {time.time() - t}")

    #sc = silhouette_score(X, labels_bis, metric="softdtw")
import pickle
pickle.dump(distortions, open("output/ts_distortions.pkl", "wb"))

plt.plot(range(2, 13), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
 """

# Setting the algorithm as the cluster with the elbow in the curve (from the above loop, it was 3)
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
mh_historical = pd.merge(mh_historical, to_merge, on= "FIPS", how = "left")
mh_historical.to_csv("output/mobile_health_historical_w_clusters.csv", index = False)

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
col_dict = {'percent_65plus':'% Aged 65 +', 'percent_opioid_deaths':'% Opioid Deaths',
       'physician_ratio':'Physician-Patient Ratio', 'percent_poor_health':'% in Poor Health', 
       'percent_rural':'% Rural', 'percent_no_health_insurance':'% with No Health Insurance',
       'percent_nonwhite':'% Non-White', 'percent_limited_english':'% with Limited English Skills', 
       'percent_no_vehicles':'% without a Vehicle', 'percent_veterans': '% Veterans',
       'percent_public_transit':'% Using Public Transit', 'percent_disabled':'% Disabled'}

writer = pd.ExcelWriter("output/clustering_results.xlsx")
for col in col_dict.keys():
    title = col_dict[col]
    groupedDF = mh_historical[["label", col]].groupby("label").describe()
    groupedDF.to_excel(writer, sheet_name = title)
writer.save()

# Adding in analysis of COVID data - how well do cases and deaths predict the time-sensitive cluster?
    # are covid cases indeed higher in the areas with higher comorbidities?
    # how good is the cluster at predicting covid cases or deaths?
covid = pd.read_csv("../severity-data/input/covid-us-counties.csv", dtype = str)
pop = pd.read_csv("input/cc-est2019-alldata.csv", dtype = str, encoding = "ISO-8859-1")
pop["FIPS"] = pop["STATE"] + pop["COUNTY"]
pop = pop.loc[(pop["YEAR"] == "12") & (pop["AGEGRP"] == "0")]
covid["cases"] = covid["cases"].astype(float)
covid["deaths"] = covid["deaths"].astype(float)

covid_grpd = covid.groupby(["fips"]).agg(cases_daily_min=("cases", min), cases_daily_max=("cases", max), cases_median=("cases", np.median), deaths_daily_min=("deaths", min), deaths_daily_max=("deaths", max), deaths_median=("deaths", np.median)).reset_index()
covid_grpd = covid_grpd.merge(pop[["FIPS", "TOT_POP"]], left_on = "fips", right_on = "FIPS", how = "left")
for col in ['cases_daily_min', 'cases_daily_max', 'cases_median', \
       'deaths_daily_min', 'deaths_daily_max', 'deaths_median']:
    covid_grpd[col + "_per100k"] = 100000 * (covid_grpd[col].astype(float) / covid_grpd["TOT_POP"].astype(float))
    covid_grpd.drop(columns = col, inplace = True)
covid_grpd.drop(columns = ["FIPS", "TOT_POP"], inplace = True)
covid_grpd = covid_grpd.loc[covid_grpd["cases_daily_min_per100k"].isnull() == False]

mh_historical = mh_historical.merge(covid_grpd, left_on = "FIPS", right_on = "fips", how = "left")
pred_narrow = mh_historical[["FIPS", "label"] + [i for i in mh_historical.columns if "per100k" in i] ].drop_duplicates()

desc = pred_narrow.groupby("label").describe().unstack(1)
desc.to_csv("output/covid_metrics_by_cluster.csv")

