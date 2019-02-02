# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:08:21 2019

@author: Jake Welch
"""

import pandas as pd

data = pd.read_csv('Incidents_Responded_to_by_Fire_Companies.csv')

#See which columns have nulls
#IM_INCIDENT_KEY, INCIDENT_TYPE_DESC, INCIDENT_DATE_TIME, ACTION_TAKEN1_DESC,
#    and BOROUGH_DESC are not missing any data 
a = data.isnull().sum()

#Q1
#Proportion of the most common type of incident
#IM_INCIDENT_KEY used for summation since none are missing 
q1 = data.groupby('INCIDENT_TYPE_DESC').count().sort_values(by='IM_INCIDENT_KEY')
q1_ans = q1['IM_INCIDENT_KEY']/q1['IM_INCIDENT_KEY'].sum()

#Q2
#Ratio of aver. # of units arriving in 111 to 651
q2a = data[data.INCIDENT_TYPE_DESC=='111 - Building fire']
q2b = data[data.INCIDENT_TYPE_DESC=='651 - Smoke scare, odor of smoke']

#Check for 0 units on scene and other outliers
#Both have at least one high value, but such is the problem of the average
q2a['UNITS_ONSCENE'].describe()
q2b['UNITS_ONSCENE'].describe()

q2_ans = q2a['UNITS_ONSCENE'].mean()/q2b['UNITS_ONSCENE'].mean()

#Q3
#Ratio of false call rate in Staten Island to false call rate in Manhattan
q3a = data[data.BOROUGH_DESC=='3 - Staten Island'].groupby(by='INCIDENT_TYPE_DESC').count()
q3b = data[data.BOROUGH_DESC=='1 - Manhattan'].groupby(by='INCIDENT_TYPE_DESC').count()

q3_top = q3a['IM_INCIDENT_KEY']['710 - Malicious, mischievous false call, other']/q3a['IM_INCIDENT_KEY'].sum()
q3_bottom = q3b['IM_INCIDENT_KEY']['710 - Malicious, mischievous false call, other']/q3b['IM_INCIDENT_KEY'].sum()

q3_ans = q3_top/q3_bottom

#Q4
#Third quartile of the distribution of minutes between logging and scene arrive for 111
q4 = data[data.INCIDENT_TYPE_DESC=='111 - Building fire']
q4t = pd.to_datetime(q4['ARRIVAL_DATE_TIME']) - pd.to_datetime(q4['INCIDENT_DATE_TIME'])
q4_ans = q4t.astype(dtype='timedelta64[s]')/60
q4_ans = q4_ans.quantile(q=0.75)

#Q5
#Find the proportion of the cooking fires that occur at the worst hour for such fires
q5 = data[data.INCIDENT_TYPE_DESC=='113 - Cooking fire, confined to container']
q5_h = pd.to_datetime(q5['INCIDENT_DATE_TIME']).dt.hour.value_counts()
q5_ans = q5_h.max()/q5_h.sum()

#Q6
#Find the R^2 value for # of residents in each zip code based on # of 111 in the zip code
#Need to use 2010 Census data for populations by zip code 
#Ignore zip codes not in the Census table, so we lose about 80 rows 
import statsmodels.regression.linear_model as sm

census_data = pd.read_csv('2010_Census_Data.csv')
q6 = data[data.INCIDENT_TYPE_DESC=='111 - Building fire']
q6['ZC'] = q6['ZIP_CODE'].astype('int')
q6_g = q6.groupby('ZC').count()
q6_g['ZIP'] = q6_g.index

q6_merge = pd.merge(q6_g,census_data,how='inner',left_on='ZIP',right_on='Zip Code ZCTA')
q6_x = q6_merge['2010 Census Population'].tolist()
q6_y = q6_merge['IM_INCIDENT_KEY'].tolist()

q6_mod = sm.OLS(q6_y,q6_x)
q6_res = q6_mod.fit()

q6_ans = q6_res.rsquared

#Q7 
#Only want data with CO detector available
#Want to compute the proportion of incidents to total type-specific incidents in the following times:
# 20-30, 30-40, 40-50, 50-60, 60-70 (including both ends in each interval; TOTAL_INC_DUR)
#i.e., Find (COdet abs in 20-30)/(COdet abs over all time) for percent/100 
#For each bin, compute COdet absent ratio/COdet present ratio 
#Perform a linear regression on the ratios to the midpoint of the bin (25,35,45,55,65)
#Predict the ratio at 39 minutes 
q7 = data[data.CO_DETECTOR_PRESENT_DESC.notnull()]
q7['TID'] = q7['TOTAL_INCIDENT_DURATION']/60

q7_ti = q7['CO_DETECTOR_PRESENT_DESC'].value_counts()

q7_23 = q7['CO_DETECTOR_PRESENT_DESC'][(q7.TID >= 20) & (q7.TID <= 30)].value_counts()
q7_34 = q7['CO_DETECTOR_PRESENT_DESC'][(q7.TID >= 30) & (q7.TID <= 40)].value_counts()
q7_45 = q7['CO_DETECTOR_PRESENT_DESC'][(q7.TID >= 40) & (q7.TID <= 50)].value_counts()
q7_56 = q7['CO_DETECTOR_PRESENT_DESC'][(q7.TID >= 50) & (q7.TID <= 60)].value_counts()
q7_67 = q7['CO_DETECTOR_PRESENT_DESC'][(q7.TID >= 60) & (q7.TID <= 70)].value_counts()

q7_df = pd.DataFrame([q7_23,q7_34,q7_45,q7_56,q7_67],index=['25','35','45','55','65'])
q7_df['Yes_Ratio'] = q7_df['Yes']/q7_ti['Yes']
q7_df['No_Ratio'] = q7_df['No']/q7_ti['No']
q7_df['Abs_to_Pres_Ratio']= q7_df['No_Ratio']/q7_df['Yes_Ratio']
q7_x = [ int(x) for x in q7_df.index]

q7_mod = sm.OLS(q7_df['Abs_to_Pres_Ratio'],q7_x)
q7_res = q7_mod.fit()
q7_ans = q7_res.predict(39)


#Q8 
#Calculate the chi-square test for whether an incident is more likely to last
#longer than 60 minutes when a CO2 test is not present 
#I interpreted this as using all data points, so a 2x2 grid of have/not CO detector
#   and greater than 60/less than or equal to 60 minutes 
q8 = data[data.CO_DETECTOR_PRESENT_DESC.notnull()].copy()
q8['TID'] = q8['TOTAL_INCIDENT_DURATION']/60

q8_gt = q8[q8.TID > 60]
q8_lt = q8[q8.TID <= 60]

q8_a = q8_gt['CO_DETECTOR_PRESENT_DESC'].value_counts().copy()
q8_b = q8_lt['CO_DETECTOR_PRESENT_DESC'].value_counts().copy()

q8_tab = pd.DataFrame(q8_a).rename(columns={'CO_DETECTOR_PRESENT_DESC':'GT'})
q8_tab['LT'] = q8_b.copy()

#DoF = (r - 1)(c - 1) = 1
#X^2 = SUM( (O-E)^2/E ), where E = (row sum)*(column sum)/(total sum)
q8_sum = q8_tab.sum().sum()
q8_gt_sum = q8_tab['GT'].sum()
q8_lt_sum = q8_tab['LT'].sum()
q8_y_sum = q8_tab.loc[q8_tab.index == 'Yes'].sum().sum()
q8_n_sum = q8_tab.loc[q8_tab.index == 'No'].sum().sum()

q8_exp = q8_tab.copy()
q8_exp['GT']['No'] = q8_gt_sum*q8_n_sum
q8_exp['GT']['Yes'] = q8_gt_sum*q8_y_sum
q8_exp['LT']['No'] = q8_lt_sum*q8_n_sum
q8_exp['LT']['Yes'] = q8_lt_sum*q8_y_sum
q8_exp = q8_exp/q8_sum

q8_diff = (q8_tab - q8_exp).pow(2)
q8_fin = q8_diff/q8_exp
q8_ans = q8_fin.sum().sum()