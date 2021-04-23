import pandas as pd
import os
import re
import numpy as np

pd.set_option('display.max_columns', 100)
os.chdir("economic-harm-data/input")

#--------------------------% living in poverty
saipe = pd.read_csv('SAIPESNC.csv')
saipe = saipe.loc[saipe["State / County Name"] != "United States"]
saipe["FIPS"] = saipe["County ID"].astype(str).apply(lambda x: x if len(x) == 5 else (5-len(x))*"0" + x)

poverty = saipe[["FIPS", "Year", "All Ages in Poverty Percent"]]
poverty.columns = ["FIPS", "year", "pct_in_poverty"]
poverty["year"] = poverty["year"].astype(str)

#-------------------------- median household income
median_income = saipe[["FIPS", "Year", "Median Household Income in Dollars"]]
median_income.columns = ["FIPS", "year", "median_income"]

#-------------------------- no college degree 25+
no_college = pd.DataFrame()
files = [i for i in os.listdir("acs_tableS1501_educational_attainment") if "data_with_overlays" in i]
for item in files:
    temp = pd.read_csv("acs_tableS1501_educational_attainment/" + item)
    temp = temp[["GEO_ID","S1501_C01_006E", "S1501_C01_012E", "S1501_C01_013E"]]
    temp = temp.iloc[1:temp.shape[0]]
    temp["year"] = item.split(".")[0][-4:]
    no_college = no_college.append(temp, ignore_index = True)
del item, files, temp

no_college["year"] = no_college["year"].astype(int)
no_college["FIPS"] = no_college["GEO_ID"].apply(lambda x: x[-5:])
no_college["no_coll_degree_25plus"] = no_college["S1501_C01_006E"].astype(float) - no_college["S1501_C01_012E"].astype(float) - no_college["S1501_C01_013E"].astype(float)
no_college["pct_no_college"] = no_college["no_coll_degree_25plus"] / no_college["S1501_C01_006E"].astype(float)
no_college = no_college[["FIPS", "year", "pct_no_college"]]

#--------------------------% part-time workers***
files = [i for i in os.listdir("acs_tableS2303_worker_status") if "data_with_overlays" in i]
worker_status = pd.DataFrame()
for item in files:
    temp = pd.read_csv("acs_tableS2303_worker_status/" + item)
    temp = temp.iloc[1:temp.shape[0]]
    year = int(item.split(".")[0][-4:])
    if year < 2015:
        temp = temp[["GEO_ID", "S2303_C01_001E", "S2303_C01_008E", "S2303_C01_017E"]]
        temp["non_part_time"] = temp["S2303_C01_008E"].astype(float) + temp["S2303_C01_017E"].astype(float)
        temp["part_time"] = temp["S2303_C01_001E"].astype(float) - temp["non_part_time"]
    else:
        temp = temp[["GEO_ID", "S2303_C01_001E", "S2303_C01_009E", "S2303_C01_030E"]]
        temp["non_part_time"] = temp["S2303_C01_009E"].astype(float) + temp["S2303_C01_030E"].astype(float)
        temp["part_time"] = temp["S2303_C01_001E"].astype(float) - temp["non_part_time"]
    
    temp = temp[["GEO_ID", "S2303_C01_001E", "part_time"]]
    temp["year"] = year
    worker_status = worker_status.append(temp, ignore_index = True)
del item, files, temp

worker_status["FIPS"] = worker_status["GEO_ID"].apply(lambda x: x[-5:])
worker_status["pct_part_time"] = worker_status["part_time"] / worker_status["S2303_C01_001E"].astype(float)
worker_status = worker_status[["FIPS", "year", "pct_part_time"]]

#-------------------------- % self employed
files = [i for i in os.listdir("acs_tableS2408_worker_class") if "data_with_overlays" in i]
workers = pd.DataFrame()
for item in files:
    temp = pd.read_csv("acs_tableS2408_worker_class/" + item)
    temp = temp[["GEO_ID", "S2408_C01_001E", "S2408_C01_004E",  "S2408_C01_009E"]]
    temp = temp.iloc[1:temp.shape[0]]
    temp["year"] = item.split(".")[0][-4:]
    workers = workers.append(temp, ignore_index = True)
del item, files, temp

workers["year"] = workers["year"].astype(int)
workers["FIPS"] = workers["GEO_ID"].apply(lambda x: x[-5:])
workers["self_employed"] = workers["S2408_C01_004E"].astype(float) + workers["S2408_C01_009E"].astype(float)
workers["pct_self_employed"] = workers["self_employed"] / workers["S2408_C01_001E"].astype(float)
workers = workers[["FIPS", "year", "pct_self_employed"]]

#-------------------------- Unemployed & Not in labor force but working age
bls_lau = pd.DataFrame()
files = [i for i in os.listdir("bls-lau-labor-force-estimates") if "xls" in i]
for item in files:
    temp = pd.read_excel("bls-lau-labor-force-estimates/" + item, sheet_name = 0, skiprows = 4)
    temp.columns = ["LAUS_Code", "State_Code", "County_Code", "County Name/State Abbreviation", "Year", "blank", "Labor_Force", "Employed", "Unemployed", "Unemployment_Rate"]
    temp = temp.loc[temp.State_Code.isnull() == False].drop(columns = "blank")
    bls_lau = bls_lau.append(temp, ignore_index = True)
del item, files, temp

bls_lau["FIPS"] = bls_lau["State_Code"].astype(int).astype(str).apply(lambda x: x if len(x) == 2 else "0" + x) + bls_lau["County_Code"].astype(int).astype(str).apply(lambda x: x if len(x) == 3 else (3 - len(x))*"0" + x )
bls_lau["year"] = bls_lau["Year"].astype(int)

# Load in age by pop data
files = os.listdir("age by state")
tot_pop = pd.DataFrame()
for item in files:
    temp = pd.read_csv("age by state/" + item, encoding ="ISO-8859-1")
    temp = temp[["STATE", "COUNTY", "YEAR", "POPESTIMATE", "AGE16PLUS_TOT"]]
    tot_pop = tot_pop.append(temp, ignore_index = True)
del temp, files, item

years = pd.DataFrame.from_dict(data = {"YEAR":[3,4,5,6,7,8,9,10,11,12], "year":[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]})

tot_pop = tot_pop.loc[tot_pop.YEAR > 2]
tot_pop = tot_pop.merge(years, on = "YEAR", how = "left")
tot_pop["FIPS"] = tot_pop["STATE"].astype(str).apply(lambda x: x if len(x) == 2 else "0" + x) + tot_pop["COUNTY"].astype(str).apply(lambda x: x if len(x) == 3 else (3-len(x))*"0" + x)
tot_pop =tot_pop[["FIPS", "year", "POPESTIMATE", "AGE16PLUS_TOT"]]

bls_lau = pd.merge(bls_lau, tot_pop, on = ["FIPS", "year"], how = "left")
bls_lau["not_in_labor_force"] = bls_lau["AGE16PLUS_TOT"] - bls_lau["Labor_Force"]

# Final dataset
bls_lau_for_concat = bls_lau.loc[bls_lau["not_in_labor_force"].isnull() == False, ["year", "FIPS", "AGE16PLUS_TOT", "not_in_labor_force", "Unemployed"]]

#-------------------------- GDP per capita
gdp = pd.read_csv("CAGDP9__ALL_AREAS_2001_2019.csv", encoding = "ISO-8859-1")
gdp = gdp.loc[(gdp.Description == "All industry total") & (gdp.GeoName != "United States")]
gdp = gdp[["GeoFIPS", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"]]

gdp_melted = gdp.melt(id_vars = "GeoFIPS")
gdp_melted.rename(columns = {"variable":"year", "value":"real_gdp", "GeoFIPS":"FIPS"}, inplace = True)
gdp_melted.replace(to_replace = "(NA)", value = np.nan, inplace = True)
gdp_melted["FIPS"] = gdp_melted["FIPS"].apply(lambda x: x.replace('"', ''))
gdp_melted["FIPS"] = gdp_melted["FIPS"].str.strip()
gdp_melted["year"] = gdp_melted["year"].astype(int)
gdp_melted["real_gdp"] = gdp_melted["real_gdp"].astype(float)

gdp_final = gdp_melted.merge(tot_pop, on= ["FIPS", "year"], how = "left")
gdp_final["gdp_per_capita"] = 1000 * (gdp_final["real_gdp"] / gdp_final["POPESTIMATE"])
gdp_final = gdp_final.loc[gdp_final["gdp_per_capita"].isnull() == False, ["FIPS", "year", "gdp_per_capita"]] 

#-------------------------- Population change
popchange = pd.read_csv("co-est2019-alldata.csv", encoding = "ISO-8859-1")
popchange["FIPS"] = popchange["STATE"].astype(int).astype(str).apply(lambda x: x if len(x) == 2 else "0" + x) + popchange["COUNTY"].astype(int).astype(str).apply(lambda x: x if len(x) == 3 else (3 - len(x))*"0" + x )
popchange = popchange[["FIPS"] + [i for i in popchange.columns.tolist() if "NPOPCHG" in i]]

popchange_melted = pd.melt(popchange, id_vars = "FIPS")
popchange_melted["year"] = popchange_melted["variable"].apply(lambda x: x.split("_")[-1])
popchange_final = popchange_melted.drop(columns = ["variable"]).rename(columns = {"value":"popchange"})
popchange_final["year"] = popchange_final["year"].astype(int)

#-------------------------- % in leisure/tourism/hospitality industry
files = [i for i in os.listdir("acs_tableS2403_industry_by_sex") if "data_with_overlays" in i]
industries = pd.DataFrame()
for item in files:
    temp = pd.read_csv("acs_tableS2403_industry_by_sex/" + item)
    temp = temp[["GEO_ID", "S2403_C01_001E", "S2403_C01_023E"]]
    temp = temp.iloc[1:temp.shape[0]]
    temp["year"] = item.split(".")[0][-4:]
    industries = industries.append(temp, ignore_index = True)
del item, files, temp

industries["year"] = industries["year"].astype(int)
industries["FIPS"] = industries["GEO_ID"].apply(lambda x: x[-5:])
industries["pct_recreation"] = industries["S2403_C01_023E"].astype(float) / industries["S2403_C01_001E"].astype(float)
industries = industries[["FIPS", "year", "pct_recreation"]]

####################################
# MERGE DATA
####################################

# begin merges
economic_harm_historical = poverty
for dataset in [median_income, no_college, worker_status, workers, bls_lau_for_concat, popchange_final, industries, gdp_final]:
    dataset["year"] = dataset["year"].astype(str)
    economic_harm_historical = economic_harm_historical.merge(dataset, how = "left", on = ["year", "FIPS"])

economic_harm_historical["pct_in_poverty"] = economic_harm_historical["pct_in_poverty"] / 100
economic_harm_historical["median_income"] = economic_harm_historical["median_income"].replace("\$|,", "", regex = True).astype(float)
economic_harm_historical = economic_harm_historical.loc[economic_harm_historical["gdp_per_capita"].isnull() == False]


####################################
# CALCULATE SCORE
####################################
# create severity score
# create severity score
d = economic_harm_historical.copy()
for column in economic_harm_historical.drop(columns = ["FIPS", "year"]).columns.tolist():
        d["q_"+ column] = d[column].rank(pct=True)

economic_harm_historical['score']= ((d['q_pct_in_poverty'] + \
    d['q_median_income'] + \
    d['q_pct_no_college'] + \
    d['q_pct_part_time'] + \
    d['q_pct_self_employed'] + \
    d['q_not_in_labor_force'] + \
    d['q_Unemployed'] + \
    d['q_popchange'] + \
    d['q_pct_recreation'] + \
    d['q_gdp_per_capita'] )/25)*100
del d

##############################################
# EXPORT LONGITUDINAL ECONOMIC HARM METRIC
##############################################
economic_harm_historical.drop(columns = ["AGE16PLUS_TOT"], inplace=True)
economic_harm_historical.to_csv("../output/economic_harm_historical.csv", index = False)
