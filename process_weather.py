# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:52:18 2017

@author: weig
"""

import pandas as pd
from sklearn import preprocessing


weather_raw = pd.read_csv(r'weather_data.csv')


# Replace missing value T with 0 
weather_raw['Precipition'].replace('T', 0,inplace=True)
weather_raw['New_Snow'].replace('T', 0,inplace=True)
weather_raw['Snow_Depth'].replace('T', 0,inplace=True)

# Standardize continuous columns

weather_raw['Max_Temp_Std'] = preprocessing.scale(weather_raw['Max_Temp'])
weather_raw['Min_Temp_Std'] = preprocessing.scale(weather_raw['Min_Temp'])
weather_raw['Avg_Temp_Std'] = preprocessing.scale(weather_raw['Avg_Temp'])
weather_raw['Temp_Departure_Std'] = preprocessing.scale(weather_raw['Temp_Departure'])
weather_raw['HDD_Std'] = preprocessing.scale(weather_raw['HDD'])
weather_raw['CDD_Std'] = preprocessing.scale(weather_raw['CDD'])
weather_raw['Precipition_Std'] = preprocessing.scale(weather_raw['Precipition'])
weather_raw['New_Snow_Std'] = preprocessing.scale(weather_raw['New_Snow'])
weather_raw['Snow_Depth_Std'] = preprocessing.scale(weather_raw['Snow_Depth'])


