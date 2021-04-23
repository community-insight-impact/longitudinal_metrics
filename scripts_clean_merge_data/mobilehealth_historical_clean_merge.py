# -*- coding: utf-8 -*-

"""
Author: Alexandra Saizan
Date Created: 11.8.2020
"""

    import pandas as pd
    import os
    import re
    import numpy as np

    pd.set_option('display.max_columns', 100)

    os.chdir("Documents/cvi/mobile-health-data")

####################################
# SET-UP
####################################

# Load helper datasets and define helper functions
#crosswalk = pd.read_csv("county_crosswalk.csv", dtype = str)
#crosswalk.rename(columns = {"Name":"County"}, inplace = True)

def get_years_in_range(yr_range):
    beginning_year, end_year = yr_range.split("-")
    return list(range( int(beginning_year), int(end_year) + 1 ))

def edit_fips(x):
    return "0" + x if len(x) < 5 else x


####################################
# LOAD AND CLEAN DATA
####################################

# ---------------------- age-adjusted percent opioid deaths
wonder_data = pd.read_csv("Multiple Cause of Death, 1999-2018.txt", delimiter="\t", usecols= ["County Code", "Age Adjusted Rate", "Year"])

# editorial decision: coerce all the "unreliable" values to zero, since the rates are unreliable becuase the number of deaths is nearly zero
wonder_data.loc[wonder_data["Age Adjusted Rate"] == "Unreliable", "Age Adjusted Rate"] = 0

wonder_data.rename(columns = {"County Code": "FIPS", "Year" : "year", "Age Adjusted Rate" : "percent_opioid_deaths"}, inplace = True)
wonder_data = wonder_data.loc[wonder_data["percent_opioid_deaths"].isnull() == False]

wonder_data["percent_opioid_deaths"] = pd.to_numeric(wonder_data["percent_opioid_deaths"], errors = "coerce")
wonder_data["FIPS"] = wonder_data["FIPS"].astype(str).apply(lambda x: edit_fips(x.split(".")[0]))
wonder_data["year"] = wonder_data["year"].astype(str).apply(lambda x: x.split(".")[0] )

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
over65 = over65.merge(pd.DataFrame({"YEAR": [1,2,3,4,5,6,7,8,9,10,11,12], "year": ["", "", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"] } ), on = "YEAR")

# final data cleaning
over65["percent_65plus"] = 100 * (over65["OVER_65"] / over65["TOT_POP"])
over65 = over65.loc[over65["YEAR"] > 2, ["FIPS", "year", "percent_65plus"]]


# ---------------------- county health rankings data 
                #(physicians rate + % in poor health + % rural)
# NOTE: percent rural is not present in the 2010 dataset

health_rankings = pd.DataFrame()

for thisFile in os.listdir("county-health-rankings"):
    year = int(re.findall("\d+", thisFile)[0])
    if year > 2018:
        physicians_col = "Ratio of population to primary care physicians."
    else:
        physicians_col = "Ratio of population to primary care physicians"
    
    df = pd.read_csv("county-health-rankings/" + thisFile)
    
    if year == 2010:
        df["% Rural raw value"] = np.nan
    
    df = df.loc[df["5-digit FIPS Code"].isin(["00000", "fipscode"]) == False, ["5-digit FIPS Code", physicians_col, "Poor or fair health raw value", "% Rural raw value"]]
        
    df["year"] = year
    
    df.rename(columns = {"5-digit FIPS Code": "FIPS", 
                         physicians_col :"physician_ratio",
                         "Poor or fair health raw value":"percent_poor_health", 
                         "% Rural raw value": "percent_rural"}, inplace = True)    
    
    df["FIPS"] = df["FIPS"].astype(str)
    df["year"] = df["year"].astype(str)
    df["physician_ratio"] = pd.to_numeric(df["physician_ratio"], errors = "coerce")
    df["percent_poor_health"] = pd.to_numeric(df["percent_poor_health"], errors = "coerce")
    df["percent_rural"] = pd.to_numeric(df["percent_rural"], errors = "coerce")

    health_rankings = health_rankings.append(df, ignore_index = True)
del df, thisFile, year, physicians_col


health_rankings["percent_poor_health"] = 100 * health_rankings["percent_poor_health"]
health_rankings["percent_rural"] = 100 * health_rankings["percent_rural"]


# ---------------------- % without health insurance

uninsured = pd.DataFrame()

for file in os.listdir("small-area-health-insurance-estimates"):
    year = str(re.findall("\d+", file)[0])
    
    df = pd.read_csv("small-area-health-insurance-estimates/" + file) 
    
    df["FIPS"] = df["statefips"].astype(str).apply(lambda x: x if len(x) > 1 else "0"*(2-len(str(x))) + str(x) ) + \
    df["countyfips"].astype(str).apply(lambda x: x if len(str(x)) > 2 else  "0"*(3 - len(str(x))) + str(x))

    df = df.loc[(df["agecat"] == 0) & (df["racecat"] == 0) & 
                (df["sexcat"] == 0) & (df["iprcat"] == 0) & 
                (df["countyfips"] != 0) ]
    
    df = df.loc[df["PCTUI"].isnull() == False, ["FIPS", "PCTUI"]]
    df["year"] = year
    df["PCTUI"] = df["PCTUI"].apply(pd.to_numeric, errors = "coerce")
    df.rename(columns = {"PCTUI": "percent_no_health_insurance"}, inplace = True)
    
    uninsured = uninsured.append(df, ignore_index = True)
del df, file, year


# ---------------------- ACS % nonwhite
path = os.path.join(os.getcwd(), "acs-B02001-race-by-sex")
l_files = [i for i in os.listdir(path) if ".csv" in i and "metadata" not in i]

race_acs = pd.DataFrame()
for file in l_files:
    df = pd.read_csv(path + "/" + file)
    df = df.loc[1:df.shape[0]]

    df.rename(columns = {"GEO_ID":"FIPS", "B02001_001E":"Total", "B02001_002E": "White_Alone"}, inplace = True)

    df["FIPS"] = df["FIPS"].apply(lambda x: x[9:14])
    df = df.loc[df["FIPS"] != ""]

    df["Total"] = df["Total"].astype(int)
    df["White_Alone"] = df["White_Alone"].astype(int)

    df["Nonwhite"] = df["Total"] - df["White_Alone"]
    df["percent_nonwhite"] = 100 * (df["Nonwhite"] / df["Total"])
    df = df[["FIPS", "percent_nonwhite"]]
    df["year"] = str(file.split(".")[0][-4:])
    race_acs = race_acs.append(df, ignore_index = True)
    
del df, file, path, l_files


# ---------------------- ACS % limited english speaking
path = os.path.join(os.getcwd(), "acs-S6102-limited-english-speaking")
l_files = [i for i in os.listdir(path) if ".csv" in i and "metadata" not in i]

limited_english = pd.DataFrame()
for file in l_files:
    df = pd.read_csv(path + "/" + file, \
                          usecols = ["GEO_ID", "S1602_C01_001E", "S1602_C03_001E"])
    df = df.loc[1:df.shape[0]]
    df.rename(columns = {"GEO_ID":"FIPS", "S1602_C01_001E":"Total_Households",\
                                  "S1602_C03_001E":"Number_Limited_English_Speaking"}, inplace = True)
    df["FIPS"] = df["FIPS"].apply(lambda x: x[9:14])
    df = df.loc[df["FIPS"] != ""]

    df["Total_Households"] = df["Total_Households"].astype(int)
    df["Number_Limited_English_Speaking"] = df["Number_Limited_English_Speaking"].astype(int)
    df["percent_limited_english"] = 100 * (df["Number_Limited_English_Speaking"] / df["Total_Households"])

    df = df[["FIPS", "percent_limited_english"]]
    df["year"] = str(file.split(".")[0][-4:])
    limited_english = limited_english.append(df, ignore_index = True)
    
del df, file, path, l_files


# ---------------------- ACS % household car ownership
path = os.path.join(os.getcwd(), "acs-S2504-household-car-ownership")
l_files = [i for i in os.listdir(path) if ".csv" in i and "metadata" not in i]

car_ownership = pd.DataFrame()
for file in l_files:
    year = str(file.split(".")[0][-4:])
    df = pd.read_csv(path + "/" + file)

    if year < 2015:
        df.rename(columns = {"GEO_ID":"FIPS", "S2504_C01_026E":"percent_no_vehicles"}, inplace = True)                         
        df = df.loc[1:df.shape[0]]
    else:
        df.rename(columns = {"GEO_ID":"FIPS","S2504_C01_001E":"Total_Occupied_Households",\
                                  "S2504_C01_027E":"Number_No_Vehicles"}, inplace = True)  
        df = df.loc[1:df.shape[0]]
        
        if any([re.search("\.", str(i))  for i in df["Number_No_Vehicles"].tolist()]):
            # Some of the years past 2015 have the estimate as a %, some years have it as a #
            df["percent_no_vehicles"] = df["Number_No_Vehicles"]
        else:   
            df["Total_Occupied_Households"] = df["Total_Occupied_Households"].astype(int)
            df["Number_No_Vehicles"] = df["Number_No_Vehicles"].astype(int)
            df["percent_no_vehicles"] = 100 * (df["Number_No_Vehicles"] / df["Total_Occupied_Households"])
    
    df["FIPS"] = df["FIPS"].apply(lambda x: x[9:14])
    df = df.loc[df["FIPS"] != ""]

    df = df[["FIPS", "percent_no_vehicles"]]
    df["year"] = year
    
    car_ownership = car_ownership.append(df, ignore_index = True)
del df, file, path, l_files


# ---------------------- ACS % veteran status
path = os.path.join(os.getcwd(), "acs-B21001-veteran-status")
l_files = [i for i in os.listdir(path) if ".csv" in i and "metadata" not in i]

veterans = pd.DataFrame()
for file in l_files:
    df = pd.read_csv(path + "/" + file, \
                          usecols = ["GEO_ID", "B21001_001E", "B21001_002E"])
    df = df.loc[1:df.shape[0]]
    df.rename(columns = {"GEO_ID":"FIPS", "B21001_001E":"Total_Pop",\
                                  "B21001_002E":"Number_Veterans"}, inplace = True)
    df["FIPS"] = df["FIPS"].apply(lambda x: x[9:14])
    df = df.loc[df["FIPS"] != ""]

    df["Total_Pop"] = df["Total_Pop"].astype(int)
    df["Number_Veterans"] = df["Number_Veterans"].astype(int)

    df["percent_veterans"] = 100 * (df["Number_Veterans"] / df["Total_Pop"])
    df = df[["FIPS", "percent_veterans"]]
    df["year"] = str(file.split(".")[0][-4:])
    veterans = veterans.append(df, ignore_index = True)
del df, file, path, l_files


# ---------------------- ACS % using public transportation
path = os.path.join(os.getcwd(), "acs-S0801-commuting-characteristics")
l_files = [i for i in os.listdir(path) if ".csv" in i and "metadata" not in i]

public_transport = pd.DataFrame()
for file in l_files:
    df = pd.read_csv(path + "/" + file, \
                          usecols = ["GEO_ID", "S0801_C01_001E", "S0801_C01_009E"])
    df = df.loc[1:df.shape[0]]
    df.rename(columns = {"GEO_ID":"FIPS", "S0801_C01_001E":"Total_Pop",\
                                  "S0801_C01_009E":"Public_Transport"}, inplace = True)
    df["FIPS"] = df["FIPS"].apply(lambda x: x[9:14])
    df = df.loc[df["FIPS"] != ""]

    df["percent_public_transit"] = df["Public_Transport"] 
    df = df[["FIPS", "percent_public_transit"]]
    df["year"] = str(file.split(".")[0][-4:])
    public_transport = public_transport.append(df, ignore_index = True)
del df, file, path, l_files

public_transport["percent_public_transit"] = pd.to_numeric(public_transport["percent_public_transit"], errors = "coerce")


# ---------------------- ACS % disability status***
path = os.path.join(os.getcwd(), "acs-S1810-disability-characteristics")
l_files = [i for i in os.listdir(path) if ".csv" in i and "metadata" not in i]

disability = pd.DataFrame()
for thisFile in l_files:
    df = pd.read_csv(path + "/" + thisFile, usecols = ["GEO_ID", "S1810_C03_001E"])
    df = df.loc[1:df.shape[0]]
    df.rename(columns = {"GEO_ID":"FIPS", "S1810_C03_001E":"percent_disabled"}, inplace = True)
    df["FIPS"] = df["FIPS"].apply(lambda x: x[9:14])
    df = df.loc[df["FIPS"] != ""]
    df["year"] = int(thisFile.split(".")[0][-4:])
    disability = disability.append(df, ignore_index = True)
del df, thisFile, path, l_files

disability["percent_disabled"] = pd.to_numeric(disability["percent_disabled"], errors = "coerce")


####################################
# MERGE DATA
####################################
datasets = [wonder_data, health_rankings, uninsured, race_acs, limited_english, car_ownership, veterans, public_transport, disability]

mobile_health_historical = over65.copy()
for dataset in datasets:
    mobile_health_historical = mobile_health_historical.merge(dataset, how = "outer", on = ["year", "FIPS"])

# keep the years that have all data
for column in mobile_health_historical.drop(labels = ["FIPS", "year"], axis =1).columns.tolist():
    print(column, mobile_health_historical.loc[mobile_health_historical[column].isnull() == False, "year"].unique() )
mobile_health_historical = mobile_health_historical.loc[mobile_health_historical.year.isin([2012, 2013, 2014, 2015, 2016, 2017, 2018])]

mobile_health_historical = mobile_health_historical.loc[mobile_health_historical["percent_65plus"].isnull() == False]

####################################
# CALCULATE SCORE
####################################
# create severity score
d = mobile_health_historical.copy()
for column in mobile_health_historical.drop(columns = ["FIPS", "year"]).columns.tolist():
        d["q_"+ column] = d[column].rank(pct=True)

mobile_health_historical['score']= ((3 * d['q_percent_rural'] + \
    2 * d['q_percent_no_vehicles'] + \
    2 * d['q_percent_public_transit'] + \
    3 * d['q_physician_ratio'] + \
    2 * d['q_percent_no_health_insurance'] + \
    d['q_percent_nonwhite'] + \
    2 * d['q_percent_limited_english'] + \
    d['q_percent_veterans'] + \
    2 * d['q_percent_65plus'] + \
    2 * d['q_percent_disabled'] + \
    d['q_percent_opioid_deaths'] + \
    d['q_percent_poor_health'] )/25)*100
del d
mobile_health_historical = mobile_health_historical.loc[mobile_health_historical.score.isnull() == False]



##############################################
# EXPORT LONGITUDINAL MOBILE HEALTH METRIC
##############################################
mobile_health_historical.to_csv("mobile_health_historical.csv", index = False)

