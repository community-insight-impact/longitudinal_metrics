# CVI Longitudinal Analysis 🕑

## Introduction
To measure the validity of the [three core CVI metrics](https://github.com/community-insight-impact/covid_community_vulnerability), we undertook a study of the metrics across time. Our motivation for this analysis was to discover how the metrics and their composite variables changed year-over-year. This reppository covers the analysis we completed for the severity metric. We started with the severity metric because it has the longest timespan of complete data.

We restricted the severity data to 2010-2017 to ensure no variables were missing for an entire year and omitted coronavirus data as it only exists for 2020. The file *severity_historical_clean_merge* merges the historical datasets of all the variables used to construct the severity score to create the longitudinal dataset. 

## Exploratory Analyses
We began the analysis by reviewing within- and across-severity score changes across time. We analyzed within-variable variation using one-way and non-parametric ANOVA tests.  We were also interested in the changes in relationships between variables, and modeled those relationships using correlation and predictive power matrices. We then reviewed year-over-year percent change in the severity score to assess its validity over time. For county-year observations with percent changes falling below the 10th or above the 90th percentile, we plan to conduct further research on whether public policy changes can explain the extreme change. 

The files *Severity Metric Exploratory Analysis.ipynb* contains our ANOVA testing, and you can find the percent change analyses in the *County-Level Exploration - Severity Metric -Updated.ipynb* file. 

![Longitudinal PPS Matrix](https://github.com/community-insight-impact/longitudinal_metrics/blob/main/pps.png)

To make analyzing changes across time easier, we also developed a Streamlit app to explore the analyses discussed above for each year. The deployed exploratory analysis app can be found at https://cvi-longitudinal-analysis.herokuapp.com/. The code used to create this app, as well ad generate the PPS and correlation matricies, can be found in *app.py*.

## Time Series Clustering

![Time Series Clusters by County](https://github.com/community-insight-impact/longitudinal_metrics/blob/main/clusters-map.png)

We conducted a time-sensitive k-means clustering analysis to group counties together based on patterns in their longitudinal severity scores. Standard k-means clustering is insufficient for modelling time-series data because Euclidean distance metrics used to measure within-cluster variance do not take time into account. We instead implemented a k-means algorithm with soft-dynamic time warping (soft-DTW) as the distance metric. Unlike dynamic time warping, soft-DTW is differentiable across all arguments, improving modelling performance. The code for this analysis can be found in *time-series-clustering.py*.
