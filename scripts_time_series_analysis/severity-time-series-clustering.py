import pandas as pd
import numpy as np
import os
import time
import json
import datetime as dt
from urllib.request import urlopen
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

pd.set_option('display.max_columns', 100)
os.chdir("/Users/alliesaizan/Dropbox/cvi/severity-data/")

severity_historical = pd.read_csv("output/severity_historical.csv")
severity_historical["FIPS"] = severity_historical["FIPS"].astype(str).apply(lambda x: "0" + x if len(x) < 5 else x)

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
distortions = []
for i in range(2, 13):

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
severity_historical.to_csv("output/severity_historical_with_cluster_labels.csv", index = False)

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

for col in col_dict.keys():
    title = col_dict[col]
    groupedDF = severity_historical[["label", col]].groupby("label").describe()
    groupedDF.to_excel("clustering_results.xlsx", sheet_name = title)

# Adding in analysis of COVID data - how well do cases and deaths predict the time-sensitive cluster?
    # are covid cases indeed higher in the areas with higher comorbidities?
    # how good is the cluster at predicting covid cases or deaths?
covid = pd.read_csv("input/covid-us-counties.csv", dtype = str)
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

severity_historical = severity_historical.merge(covid_grpd, left_on = "FIPS", right_on = "fips", how = "left")
pred_narrow = severity_historical[["FIPS", "label"] + [i for i in severity_historical.columns if "per100k" in i] ].drop_duplicates()

desc = pred_narrow.groupby("label").describe().unstack(1)
desc.to_csv("output/covid_metrics_by_cluster.csv")

# For the periods where counties are classified as hotspots, are those hotspots more likely to be classified as a partcular cluster?
"""
Counties defined as hotspot counties met all four of the following criteria, relative to the date assessed: 
1) >100 new COVID-19 cases in the most recent 7 days
2) an increase in the most recent 7-day COVID-19 incidence over the preceding 7-day incidence
3) a decrease of <60% or an increase in the most recent 3-day COVID-19 incidence over the preceding 3-day incidence
4) the ratio of 7-day incidence/30-day incidence exceeds 0.31
5) >60% change in the most recent 3-day COVID-19 incidence OR >60% change in the most recent 7-day incidence
"""

covid["date"] = pd.to_datetime(covid["date"])
covid = covid[["fips", "date", "cases", "deaths"]].sort_values(by = ["fips", "date"])
covid["cases_daily_change"] = covid.groupby(["fips"])["cases"].diff()
covid = covid.loc[covid.fips.isnull() == False]

hotspots = pd.DataFrame(columns = ["ID", "fips", "date", "rule_1_check", "rule_2_check", "rule_3_check", "rule_4_check"])

def find_past_x_days(current_date, num_days):
    return [current_date - dt.timedelta(days=x) for x in range(num_days)]

def find_hotspots(df, hotspots):
    for idx, row in df.iterrows():
        fips = row["fips"]
        current_date = row["date"]
        past_7_days = find_past_x_days(current_date, 7)
        past_14_days = find_past_x_days(current_date, 14)
        preceeding_7_days = list(set(past_14_days).difference(set(past_7_days)))

        past_3_days = find_past_x_days(current_date, 3)
        past_6_days = find_past_x_days(current_date, 6)
        preceeding_3_days = list(set(past_6_days).difference(set(past_3_days)))
        
        past_30_days = find_past_x_days(current_date, 30)

        hotspots = hotspots.append({"ID": idx, "fips":fips, "date":current_date}, ignore_index = True)
        
        # rule 1
        subset_rows = df.loc[(df["fips"] == fips) & (df["date"] in past_7_days) & (df["cases_daily_change"] > 100)]
        # if subset_rows.shape[0] > 0:
        #     hotspots.loc[idx, "rule_1_check"] = 1
        # else:
        #     hotspots.loc[idx, "rule_1_check"] = 0

        #  rule 2:
        mean_past_7_days = df.loc[(df["fips"] == fips) & (df["date"] in past_7_days), "cases"].mean()
        mean_preceeding_7_days = df.loc[(df["fips"] == fips) & (df["date"] in preceeding_7_days), "cases"].mean()
        if mean_past_7_days > mean_preceeding_7_days:
            hotspots.loc[idx, "rule_2_check"] = 1
        else:
            hotspots.loc[idx, "rule_2_check"] = 0
        
        # rule 3:
        mean_past_3_days = df.loc[(df["fips"] == fips) & (df["date"] in past_3_days), "cases"].mean()
        mean_preceeding_3_days = df.loc[(df["fips"] == fips) & (df["date"] in preceeding_3_days), "cases"].mean()
        if mean_past_3_days > mean_preceeding_3_days:
            hotspots.loc[idx, "rule_3_check"] = 1
        else:
            hotspots.loc[idx, "rule_3_check"] = 0

        # rule 4
        mean_past_30_days = df.loc[(df["fips"] == fips) & (df["date"] in past_30_days), "cases"].mean()
        if mean_past_7_days / mean_past_30_days > 0.31:
            hotspots.loc[idx, "rule_4_check"] = 1
        else:
            hotspots.loc[idx, "rule_4_check"] = 0

        # rule 5:
        r5_check1 = (df.at[(df["fips"] == fips) & (df["date"] == past_3_days[-1]), "cases"] - df.at[(df["fips"] == fips) & (df["date"] == past_3_days[0]), "cases"]) / df.at[(df["fips"] == fips) & (df["date"] == past_3_days[-1]), "cases"]
        r5_check2 = (df.at[(df["fips"] == fips) & (df["date"] == past_7_days[-1]), "cases"] - df.at[(df["fips"] == fips) & (df["date"] == past_7_days[0]), "cases"]) / df.at[(df["fips"] == fips) & (df["date"] == past_7_days[-1]), "cases"]
        if r5_check1 > 0.6 or r5_check2 > 0.6:
            hotspots.loc[idx, "rule_5_check"] = 1
        else:
            hotspots.loc[idx, "rule_5_check"] = 0

        return hotspots

df_new = find_hotspots(covid, hotspots = hotspots)

# %%
