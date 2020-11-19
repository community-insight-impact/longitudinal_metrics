# -*- coding: utf-8 -*-

"""
Author: Alexandra Saizan
Date Created: 11.8.2020
Summary: 

"""

import pandas as pd
import os
import re

pd.set_option('display.max_columns', 100)

os.chdir("/Users/admin/Documents/cvi/severity-data")

####################################
# SET-UP
####################################

# Load helper datasets and define helper functions

crosswalk = pd.read_csv("county_crosswalk.csv", dtype = str)
crosswalk.rename(columns = {"Name":"County"}, inplace = True)

def get_years_in_range(yr_range):
    beginning_year, end_year = yr_range.split("-")
    return list(range( int(beginning_year), int(end_year) + 1 ))


def generate_long_data(source_dat, year_ranges):
    df = pd.DataFrame()
    for bucket in year_ranges:
        ranges = get_years_in_range(bucket)
        for year in ranges:
            subset_df = source_dat.loc[source_dat["Year"] == bucket]
            subset_df["year"] = year
            df = df.append(subset_df, ignore_index = True)
    df = pd.merge(df, crosswalk, how = "left", on = ["County", "State"])
    return df


def edit_fips(x):
    return "0" + x if len(x) < 5 else x


def append_files(file_path, variable):
    resultDF = pd.DataFrame()
    for file in os.listdir(file_path): 
        tempDF = pd.read_csv(file_path + "/" + file, header = 2)
        tempDF = tempDF.loc[tempDF["CountyFIPS"].isnull() == False]
        tempDF["FIPS"] = tempDF["CountyFIPS"].astype(str).apply(lambda x: edit_fips(x.split(".")[0]))
        tempDF.rename(columns = {"Percentage": variable}, inplace = True)
        tempDF["year"] = re.findall("\d+", file)[0]
        resultDF = resultDF.append(tempDF, ignore_index = True)
    return resultDF


####################################
# LOAD AND CLEAN DATA
####################################

# ---------------------- percent of residents aged 65+
ccest2019 = pd.read_csv("cc-est2019-alldata.csv", encoding = "ISO-8859-1", sep = ",")

# data cleaning
ccest2019["FIPS"] = ccest2019.STATE.astype(str).apply(lambda x: "0" + x if len(x) < 2 else x) + \
        ccest2019.COUNTY.astype(str).apply(lambda x: "0" * (3 - len(x)) + x )

ccest2019 = ccest2019.loc[ccest2019.AGEGRP.isin([0, 14, 15, 16, 17, 18]), ["FIPS", "AGEGRP", "YEAR", "TOT_POP"]]

tot_pop = ccest2019.loc[ccest2019.AGEGRP == 0 ]
tot_pop.drop(labels = ["AGEGRP"], axis = 1, inplace = True)

over65 = ccest2019.loc[ccest2019.AGEGRP != 0 ]
over65 = over65.drop(columns = ["AGEGRP"], axis = 1).groupby(by = ["FIPS", "YEAR"]).sum().reset_index()
over65.rename(columns = {"TOT_POP":"OVER_65"}, inplace = True)

# merges
over65 = over65.merge(tot_pop, on = ["FIPS", "YEAR"])
over65 = over65.merge(pd.DataFrame({"YEAR": [1,2,3,4,5,6,7,8,9,10,11,12], \
                                            "year": ["", "", "2010", "2011", "2012", "2013", "2014", \
                                                     "2015", "2016", "2017", "2018", "2019"] } ),\
                      on = "YEAR")

# final data cleaning
over65["percent_65plus"] = 100 * (over65["OVER_65"] / over65["TOT_POP"])
over65 = over65.loc[over65["YEAR"] > 2, ["FIPS", "year", "percent_65plus"]]


# ---------------------- total cardiovascular disease death rate **
heart_disease_mortality = pd.read_csv("total_heart_diease_data.csv")

# get the unique list of year ranges
year_ranges = heart_disease_mortality["Year"].unique().tolist()

# create aggregated dataset with duplicates 
split_duplicated = generate_long_data(heart_disease_mortality, year_ranges)

# merge and finalize
heart_disease_mortality = split_duplicated.drop(columns = ["Category Range", "Year", "County", "State"], axis = 1)
heart_disease_mortality.rename(columns = {"Value": "total_heart_disease_death_rate" }, inplace = True)
heart_disease_mortality.FIPS = heart_disease_mortality.FIPS.astype(str).apply(edit_fips)
heart_disease_mortality["year"] = heart_disease_mortality["year"].astype(str)


# ---------------------- hypertension death rate **
hypertension_mortality = pd.read_csv("hypertension_historical_data.csv")

# get the unique list of year ranges
year_ranges = hypertension_mortality["Year"].unique().tolist()

# create aggregated dataset with duplicates 
split_duplicated = generate_long_data(hypertension_mortality, year_ranges)

# merge and finalize
hypertension_mortality = split_duplicated.drop(columns = ["Category Range", "Year", "County", "State"], axis = 1)
hypertension_mortality.rename(columns = {"Value": "hypertension_death_rate" }, inplace = True)
hypertension_mortality.FIPS = hypertension_mortality.FIPS.astype(str).apply(edit_fips)
hypertension_mortality["year"] = hypertension_mortality["year"].astype(str)


# ---------------------- COPD death rate **
copd_mortality = pd.read_excel("IHME_USA_COUNTY_RESP_DISEASE_MORTALITY_1980_2014_NATIONAL_Y2017M09D26.xlsx", \
                     sheet_name = 1, header = 1)
helper_dict = {"2006":"2005", "2007":"2005", "2008":"2005", "2009":"2005", \
              "2011":"2010", "2012":"2010", "2013":"2010", \
              "2015": "2014", "2016":"2014", "2017":"2014", "2018":"2014"}

# remove rows with missing FIPS codes
copd_mortality = copd_mortality.loc[copd_mortality.FIPS.isnull() == False, \
                                    ["FIPS", "Mortality Rate, 2005*", "Mortality Rate, 2010*", "Mortality Rate, 2014*"]]

# reshape data wide to long
copd_mortality_grpd = pd.melt(copd_mortality, id_vars = ["FIPS"], \
                              value_vars = ["Mortality Rate, 2005*", "Mortality Rate, 2010*", "Mortality Rate, 2014*"])

# data cleaning
copd_mortality_grpd["year"] = copd_mortality_grpd.variable.apply(lambda x: x.split(",")[1].split("*")[0].strip() )
copd_mortality_grpd = copd_mortality_grpd[["FIPS", "year", "value"]]
copd_mortality_grpd.FIPS = copd_mortality_grpd.FIPS.astype(str).apply(lambda x: x.split(".")[0] ).apply(edit_fips)
copd_mortality_grpd["copd_death_rate"] = pd.to_numeric(copd_mortality_grpd["value"].apply(lambda x: x.split(" ")[0]))
copd_mortality_grpd.drop(labels = ["value"], axis = 1, inplace = True)

# add data for missing years
for missing_year in helper_dict.keys():
    temp_df = copd_mortality_grpd.loc[copd_mortality_grpd["year"] == helper_dict[missing_year] ]
    temp_df["year"] = missing_year
    copd_mortality_grpd = copd_mortality_grpd.append(temp_df, ignore_index = True)
del missing_year, temp_df


# ---------------------- percent of residents who are smokers
smokers = pd.DataFrame()

for file in os.listdir("county-health-rankings"):
    tempDF = pd.read_csv(os.getcwd() + "/county-health-rankings/"  + file, skiprows=([1]))
    tempDF = tempDF.loc[tempDF["Adult smoking raw value"].isnull() == False,
                        ["5-digit FIPS Code", "Adult smoking raw value"]]
    tempDF["year"] = re.findall("\d{4}", file)[0]
    tempDF.rename(columns = {"5-digit FIPS Code": "FIPS", "Adult smoking raw value":"percent_smokers"}, inplace = True)
    smokers = smokers.append(tempDF, ignore_index = True)    
del tempDF, file

smokers["FIPS"] = smokers.FIPS.astype(str).apply(lambda x: edit_fips(x.split(".")[0]))
smokers["percent_smokers"] = 100 * smokers["percent_smokers"] 

# ---------------------- percent of residents diagnosed with diabetes
diabetes = append_files(file_path = "diabetes", variable = "percent_diabetes")
diabetes = diabetes[["FIPS", "percent_diabetes", "year"]]

# ---------------------- percent of residents diagnosed with obesity

obesity = append_files(file_path = "obesity", variable = "percentt_obese")
obesity = obesity[["FIPS", "percent_obese", "year"]]

####################################
# MERGE DATA
####################################

# begin merges
severity_historical = over65
for dataset in [smokers, diabetes, obesity, copd_mortality_grpd, hypertension_mortality, heart_disease_mortality]:
    severity_historical = severity_historical.merge(dataset, how = "left", on = ["year", "FIPS"])


####################################
# EXPORT LONGITUDINAL SEVERITY METRIC
####################################

severity_historical.to_csv("severity_historical.csv", index = False)