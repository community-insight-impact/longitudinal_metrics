# -*- coding: utf-8 -*-

"""
Author: Alexandra Saizan
Date Created: 11.8.2020
"""

import pandas as pd
import os
from scipy.stats import f_oneway, kruskal
import seaborn as sns
from matplotlib import pyplot as plt
import ppscore as pps
from urllib.request import urlopen
import json
import plotly.express as px

pd.set_option('display.max_columns', 100)
os.chdir("/Users/alliesaizan/Dropbox/cvi/severity-data/")

####################################
# INITIAL REVIEW
####################################

severity_historical = pd.read_csv("output/severity_historical.csv")
crosswalk = pd.read_csv("input/county_crosswalk.csv")

correlations = severity_historical.corr()
correlations.to_csv("output/severity_historical_correlations.csv")

####################################
# ONE-WAY ANOVA
####################################
    # The one-way ANOVA tests the null hypothesis that two or more groups 
    # have the same population mean. The test is applied to samples from 
    # two or more groups, possibly with differing sizes. I will calculate the 
    # ANOVA for each variable across time periods.
    # A p-value of 0.05 tell us that there is a less than 5% probability than the difference in means is observed due to random chance; in other words, the difference is statistically significant.
    # The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. This nonparametric method for is used for samples that do not satisfy the normality assumption. The p-value returned uses the assumption that H has a chi square distribution.

def generate_anova(df, testname):
    for colname in df.drop(labels = ["FIPS", "year"], axis = 1).columns.tolist():
        df_new = df.loc[df[colname].isnull() == False, ["year", colname]]
        for_anova = [df_new.loc[df["year"] == year, colname].tolist() for year in df.year.unique()]
        if testname == "Kruskal-Wallis H-test":
            result = kruskal(for_anova[0], for_anova[1], for_anova[2], for_anova[3], for_anova[4], for_anova[5], for_anova[6])
        else:
            result = f_oneway(for_anova[0], for_anova[1], for_anova[2], for_anova[3], for_anova[4], for_anova[5], for_anova[6])
        txt = "{name} for {col}: ".format(col = colname, name=testname)
        print(txt + str(result) + "\n")
        #with open("output/mobile_health_anova_results.txt") as f:
        #     f.write(txt + str(result))

generate_anova(severity_historical, "Oneway ANOVA")
generate_anova(severity_historical, "Kruskal-Wallis H-test")


####################################
# PREDCTIVE POWER SCORE
####################################

subsetdf = severity_historical.loc[severity_historical["year"].isin(["2010", "2011", "2012", "2013", "2013", "2014", "2015", "2016", "2017"])].drop(labels = ["FIPS", "year"], axis = 1)
subsetdf = subsetdf.rename(columns = {"percent_65plus":"% 65 Plus", "percent_smokers": "% Smokers", "percent_diabetes":"% With Diabetes", "percent_obese":"% Obese", "copd_death_rate":"COPD*", "hypertension_death_rate":"Hypertension*", "total_heart_disease_death_rate":"Total Heart Disease*", "severe_cases":"Severe Cases"})
subsetdf.drop(columns = ["Severe Cases"], axis=1, inplace = True)
matrix_df = pps.matrix(subsetdf)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

#colors = ["#FFFFFF", "#01AFB8", "#196E9F"]
palette = sns.light_palette("#196E9F", as_cmap=True)

fig = plt.figure()
heatmap = sns.heatmap(matrix_df, cmap = palette, annot=True)
heatmap.set(xlabel=None, ylabel=None)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)

plt.subplots_adjust(bottom=0.5)
fig.savefig("output/severity_pps.png")

####################################
# PERCENT CHANGE ANALYSIS
####################################
# Calculate percent change in sever cases by year for each county
severity_grpd = severity_historical[["FIPS", "year", "severe_cases"]].sort_values(by = ["FIPS", "year"])
severity_grpd["pct_change"] = severity_grpd.groupby(["FIPS"])["severe_cases"].pct_change()

# Calculate percentiles for the percent change metric (note: team input here?)
percentiles = severity_grpd["pct_change"].quantile(q = [0.1, 0.9], interpolation = "midpoint")
severity_percentiles = severity_grpd.loc[(severity_grpd["pct_change"] < percentiles[0.1]) | (severity_grpd["pct_change"] > percentiles[0.9]) ]

# Check the duplicates by FIPS code
severity_percentiles.pivot_table(index=['FIPS'], aggfunc='size')

# Top & Bottom 20 counties by quantile
severity_percentiles.sort_values(by = "pct_change", inplace = True) 

print("Smallest percent changes: " + str(severity_percentiles.iloc[0:20, 0].tolist()) + "\n\nLargest percent changes: " + str(severity_percentiles.iloc[-20:, 0].tolist()) )

# Calculate percent change in sever cases by year for each county
severity_grpd = severity_grpd.merge(crosswalk, on = "FIPS", how = "left")

""" # Map it!
fig = px.choropleth(severity_percentiles[["FIPS", "pct_change"]], geojson=counties, locations='FIPS', color='pct_change',
                           color_continuous_scale=palette,
                           range_color=(0, 12),
                           scope="usa",
                           labels={'pct_change':'Extreme percent changes'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show() """


with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


###########################
# Mean percent changes

severity_grpd_mean = severity_grpd.groupby(["FIPS"]).agg({"pct_change":["mean"]}).reset_index()
severity_grpd_mean.columns = ["FIPS","mean_pct_change"]
palette = sns.light_palette("#196E9F", as_cmap=True)

fig = px.choropleth(severity_grpd_mean, geojson=counties, locations='FIPS', color='mean_pct_change',
                           color_continuous_scale=["#FFFFFF", "#196E9F"],
                           scope="usa",
                           labels={'mean_pct_change':'Mean Percent Change'} )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image("output/severity_mean_pct_change_map.png")