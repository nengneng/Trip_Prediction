# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:58:59 2017

@author: ds1
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:07:56 2017

@author: nadiao
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Train = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train.csv')
Test = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\test.csv')

'''
Time: 
    - Year, Month, Day, Hour, Minute, Second, 
    - DayOfWeek (Monday = 0, Sunday = 6), IsWeekend, IsHoliday (1 day before and after), Season, IsRushHour
    
    - need to do: Season?, IsRushHour
'''
#for train: feature engineering for Time
Train['start_datetime'] = Train['start_timestamp'].astype('int').astype("datetime64[s]")

Train['start_year'], Train['start_month'], Train['start_day'], Train['start_date'], Train['start_dayOfWeek'], Train['start_time'], Train['start_hour'], Train['start_minute'],  Train['start_second'] = Train['start_datetime'].dt.year, Train['start_datetime'].dt.month, Train['start_datetime'].dt.day, Train['start_datetime'].dt.date, Train['start_datetime'].dt.dayofweek, Train['start_datetime'].dt.time, Train['start_datetime'].dt.hour, Train['start_datetime'].dt.minute, Train['start_datetime'].dt.second

Train.loc[Train['start_dayOfWeek'].isin([5, 6]), 'start_isWeekend'] = 1
Train.loc[Train['start_dayOfWeek'].isin([0, 1, 2, 3, 4]), 'start_isWeekend'] = 0

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
holidays = cal.holidays(start=Train['start_date'].min(), end=Train['start_date'].max())
Train['start_isHoliday'] = np.where(Train.start_datetime.dt.normalize().isin(holidays), 1, 0)

#for train: time features description

#daily counts of 
TimeSeriesTrain = pd.DataFrame(Train['start_date'].value_counts().reset_index().rename(columns={'index': 'start_date', 0: 'Count'}))
TimeSeriesTrain.columns = ['start_date', 'Count']
TimeSeriesTrain = TimeSeriesTrain.sort_values(by='start_date', ascending=True)

plt.figure(figsize=(30,10))
plt.rcParams.update({'font.size': 22})
plt.plot(TimeSeriesTrain['start_date'], TimeSeriesTrain['Count'])
plt.tight_layout()

TimeSeriesTrain['Count'].describe()

TimeSeriesTrain.loc[TimeSeriesTrain['Count'] == 6728]['start_date']  # 2015-01-27
TimeSeriesTrain.loc[TimeSeriesTrain['Count'] == 45666]['start_date']  # 2015-01-31

Train['start_year'].value_counts()/Train['start_year'].value_counts().sum()

Train['start_month'].value_counts()/Train['start_month'].value_counts().sum()

Train['start_dayOfWeek'].value_counts()/Train['start_dayOfWeek'].value_counts().sum()

Train['start_isWeekend'].value_counts()/Train['start_isWeekend'].value_counts().sum()

Train['start_hour'].value_counts()/Train['start_hour'].value_counts().sum()





#for test
Test['start_datetime'] = Test['start_timestamp'].astype('int').astype("datetime64[s]")

Test['start_year'], Test['start_month'], Test['start_day'], Test['start_date'], Test['start_dayOfWeek'], Test['start_time'], Test['start_hour'], Test['start_minute'],  Test['start_second'] = Test['start_datetime'].dt.year, Test['start_datetime'].dt.month, Test['start_datetime'].dt.day, Test['start_datetime'].dt.date, Test['start_datetime'].dt.dayofweek, Test['start_datetime'].dt.time, Test['start_datetime'].dt.hour, Test['start_datetime'].dt.minute, Test['start_datetime'].dt.second

Test.loc[Test['start_dayOfWeek'].isin([5, 6]), 'start_isWeekend'] = 1
Test.loc[Test['start_dayOfWeek'].isin([0, 1, 2, 3, 4]), 'start_isWeekend'] = 0

Test['start_isHoliday'] = np.where(Test.start_datetime.dt.normalize().isin(holidays), 1, 0)

#for Test: time features description
 
#daily counts of 
TimeSeriesTest = pd.DataFrame(Test['start_date'].value_counts().reset_index().rename(columns={'index': 'start_date', 0: 'Count'}))
TimeSeriesTest.columns = ['start_date', 'Count']
TimeSeriesTest = TimeSeriesTest.sort_values(by='start_date', ascending=True)

plt.figure(figsize=(30,10))
plt.rcParams.update({'font.size': 22})
plt.plot(TimeSeriesTest['start_date'], TimeSeriesTest['Count'])
plt.tight_layout()

TimeSeriesTest['Count'].describe()

TimeSeriesTest.loc[TimeSeriesTest['Count'] == 694]['start_date']  # 2015-01-27
TimeSeriesTest.loc[TimeSeriesTest['Count'] == 5086]['start_date']  # 2015-05-03

Test['start_year'].value_counts()/Test['start_year'].value_counts().sum()

Test['start_month'].value_counts()/Test['start_month'].value_counts().sum()

Test['start_dayOfWeek'].value_counts()/Test['start_dayOfWeek'].value_counts().sum()

Test['start_isWeekend'].value_counts()/Test['start_isWeekend'].value_counts().sum()

Test['start_hour'].value_counts()/Test['start_hour'].value_counts().sum()

'''
Duration description
'''

Train.duration.describe()


plt.hist(Train.duration, normed=True, bins=50)

removedExtremeDurations = Train.loc[Train['duration'] < 2000]
len(removedExtremeDurations)/len(Train) #0.7150309765867292
plt.hist(removedExtremeDurations.duration, normed=True, bins=50)

onlyExtremeDurations = Train.loc[Train['duration'] >= 1000]
len(onlyExtremeDurations)/len(Train) #0.2849690234132708
plt.hist(onlyExtremeDurations.duration, range=[1000, 11000], normed=True, bins=10)

durationTimeSeriesAvg = Train['duration'].groupby(Train['start_date']).mean().to_frame()
durationTimeSeriesMax = Train['duration'].groupby(Train['start_date']).max().to_frame()
durationTimeSeriesMin = Train['duration'].groupby(Train['start_date']).min().to_frame()

durationTimeSeries = pd.concat([durationTimeSeriesAvg, durationTimeSeriesMax, durationTimeSeriesMin], axis=1, join_axes=[durationTimeSeriesAvg.index])
durationTimeSeries.columns = ['mean', 'max', 'min']

plt.plot(durationTimeSeriesAvg)
plt.plot(durationTimeSeriesMax)
plt.plot(durationTimeSeriesMin)


''' 
Distance: 
    haversineDist, ArchDist, ManhattonDist, EuclidDist
'''
#for train
import math
def distance(lat1, lon1, lat2, lon2):
    #radius = 6371 # km
    radius = 3959 #miles

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

Trainlat1 = Train['start_lat'].tolist()
Trainlon1 = Train['start_lng'].tolist()
Trainlat2 = Train['end_lat'].tolist()
Trainlon2 = Train['end_lng'].tolist()

tripDistance_Train = list(map(distance, Trainlat1, Trainlon1, Trainlat2, Trainlon2))

tripDistance_TrainDF = pd.DataFrame({'haversineDist': tripDistance_Train})

Train = pd.concat([Train, tripDistance_TrainDF], axis=1)

plt.hist(Train.haversineDist, normed=True, bins=100)
Train.haversineDist.describe()

#for test
Testlat1 = Test['start_lat'].tolist()
Testlon1 = Test['start_lng'].tolist()
Testlat2 = Test['end_lat'].tolist()
Testlon2 = Test['end_lng'].tolist()

tripDistance_Test = list(map(distance, Testlat1, Testlon1, Testlat2, Testlon2))

tripDistance_TestDF = pd.DataFrame({'haversineDist': tripDistance_Test})

Test = pd.concat([Test, tripDistance_TestDF], axis=1)

plt.hist(Test.haversineDist, normed=True, bins=100)
Test.haversineDist.describe()

'''
Location: 
    - Categorized Long/Lat


    https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature
    https://datascience.stackexchange.com/questions/23651/can-gps-coordinates-latitude-and-longitude-be-used-as-features-in-a-linear-mod
    - GeoHash: 
            http://www.movable-type.co.uk/scripts/geohash.html
'''

#Create Geohash for train
import pygeohash as pgh

Train_start_geohash = [pgh.encode(x,y) for x,y in zip(Trainlat1, Trainlon1)]
Train_end_geohash = [pgh.encode(x,y) for x,y in zip(Trainlat2, Trainlon2)]
Train_geodistance = [pgh.geohash_approximate_distance(x,y) for x,y in zip(Train_start_geohash, Train_end_geohash)]

len(Train_start_geohash) #12905715
len(set(Train_start_geohash)) #9753342
len(Train_end_geohash) #12905715
len(set(Train_end_geohash)) #10832199

len(Train_geodistance) #12905715
len(set(Train_geodistance)) #9  


start_geohash_TrainDF = pd.DataFrame({'start_geohash': Train_start_geohash})
end_geohash_TrainDF = pd.DataFrame({'end_geohash': Train_end_geohash})
geodistance_TrainDF = pd.DataFrame({'geoDist': Train_geodistance})

Train = pd.concat([Train, geodistance_TrainDF], axis=1)
Train = pd.concat([Train, start_geohash_TrainDF], axis=1)
Train = pd.concat([Train, end_geohash_TrainDF], axis=1)

plt.hist(Train.geoDist, normed=True, bins=100)
Train.geoDist.describe()

Train.geoDist.value_counts() #why geoDist is just these specific ones??

'''
Remove outliers:
    - Super long trips >2000 seconds
    - Super short trips < 60 seconds

'''
removedExtremeDurations = Train.loc[Train['duration'] < 1500]
len(removedExtremeDurations)/len(Train) #0.9434903839113137
plt.hist(removedExtremeDurations.duration, normed=True, bins=50)

removedExtremeDurations2 = removedExtremeDurations.loc[removedExtremeDurations['duration'] > 60]
len(removedExtremeDurations2)/len(Train) #0.9376480884631344

OldTrain = Train
Train = removedExtremeDurations2.reset_index(drop = True)
Train.shape
removedExtremeDurations2.shape



'''
Location
    - Get center center lat, lng
    - Get start distance to city center
    - Get HeadingCityCenter

'''
Train['end_lng'].median #-74.004326
Train['end_lat'].median # 40.719986
CityCenter_lng_Train_ls = [-74.004326] * 12101019
CityCenter_lat_Train_ls = [40.719986] * 12101019
CityCenter_train = pd.DataFrame({'CityCenter_lng': CityCenter_lng_Train_ls, 'CityCenter_lat': CityCenter_lat_Train_ls})

Train = Train.join(CityCenter_train)


#distance to City Center from start
TrainStartlat = Train['start_lat'].tolist()
TrainStartlon = Train['start_lng'].tolist()
TrainCityCenterlat = Train['CityCenter_lat'].tolist()
TrainCityCenterlon = Train['CityCenter_lng'].tolist()

DistanceToCityCenter_Train = list(map(distance, TrainStartlat, TrainStartlon, TrainCityCenterlat, TrainCityCenterlon))

DistanceToCityCenter_TrainDF = pd.DataFrame({'DistanceStartToCityCenter': DistanceToCityCenter_Train})

Train = pd.concat([Train, DistanceToCityCenter_TrainDF], axis=1)

#heading to CityCenter
def heading(lat1, lng1, lat2, lng2):
    
    lat1, lng1 = math.radians(lat1), math.radians(lng1)
    lat2, lng2 = math.radians(lat2), math.radians(lng2)
    aa = np.sin(lng2 - lng1) * np.cos(lat2)
    bb = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng2 - lng1)
    return np.arctan2(aa, bb) + np.pi 

HeadingToCityCenter_Train = list(map(heading, TrainStartlat, TrainStartlon, TrainCityCenterlat, TrainCityCenterlon))

HeadingToCityCenter_TrainDF = pd.DataFrame({'HeadingToCityCenter': HeadingToCityCenter_Train})

Train = pd.concat([Train, HeadingToCityCenter_TrainDF], axis=1)


'''
External Weather API call: 
    WeatherStartLoc_StartTime, WeatherEndLoc_StartTime, 

Other ideas to consider: 
    - driver
        age, years of driving experience, years of driving experience in current city, avg driving speed-highway/local, driver ratings, #cars for this driver 
    - car
        age, make, accidents
    - passenger
        number of passengers, passenger gender/age, passenger tourist/local
    - city specific model
'''



'''

Models to consider:
    - linear model + Regularization
    - GAM
    - Random Forest
    - XGBoost
    - Neural Nets

Special Model:
    - convert continuous Y to multi-class labels
    - build classification model
    - when scoring, provide predicted bucket, mean seconds in each bucket 
    
Cross-Validation:
    -Mean Absolute Prediction Error

Model-persistence
    - pickle model
    - save model
    - call model
    
Data Exploration Charts
    - Univariate x-y
    - Heat Map for longitude/latitude
    - 
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Train.to_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv',index=False)
#Train.to_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv',index=False)

Train = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv')
#Test = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\test_clean.csv')
Train.isnull().values.any()
Train.head()

import sklearn
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
 
#some correlation plots
plt.scatter(Train['haversineDist'], Train['duration'])
plt.xlabel('haversineDist')
plt.ylabel('duration')
plt.title('haversineDist vs duration')
plt.show()

plt.scatter(Train['DistanceStartToCityCenter'], Train['duration'])
plt.xlabel('DistanceStartToCityCenter')
plt.ylabel('duration')
plt.title('DistanceStartToCityCenter vs duration')
plt.show()

plt.scatter(Train['HeadingToCityCenter'], Train['duration'])
plt.xlabel('HeadingToCityCenter')
plt.ylabel('duration')
plt.title('HeadingToCityCenter vs duration')
plt.show()

plt.scatter(Train['geoDist'], Train['duration'])
plt.xlabel('geoDist')
plt.ylabel('duration')
plt.title('geoDist vs duration')
plt.show()
   
plt.scatter(Train['start_hour'], Train['duration'])
plt.xlabel('start_hour')
plt.ylabel('duration')
plt.title('start_hour vs duration')
plt.show()
   
plt.scatter(Train['start_isWeekend'], Train['duration'])
plt.xlabel('start_isWeekend')
plt.ylabel('duration')
plt.title('start_isWeekend vs duration')
plt.show()
   
plt.scatter(Train['start_isHoliday'], Train['duration'])
plt.xlabel('start_isHoliday')
plt.ylabel('duration')
plt.title('start_isHoliday vs duration')
plt.show()


durationVSstart_dayOfWeek = removedExtremeDurations['duration'].groupby(removedExtremeDurations['start_dayOfWeek']).mean().to_frame()
plt.plot(durationVSstart_dayOfWeek)

durationVSstart_hour = removedExtremeDurations['duration'].groupby(removedExtremeDurations['start_hour']).mean().to_frame()
plt.plot(durationVSstart_hour)

durationVSstart_hour = removedExtremeDurations['duration'].groupby(removedExtremeDurations['start_hour']).mean().to_frame()
plt.plot(durationVSstart_hour)

#did not use features: start_date, start_time

#transform dytpes:
#TrainFeatures = Train[['start_year', 'start_month', 'start_day',  'start_dayOfWeek',  'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday','haversineDist', 'geoDist', 'start_geohash', 'end_geohash']]
#            
#TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'start_geohash', 'end_geohash']] = TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'start_geohash', 'end_geohash']].astype(object)

'''
First round, duration as target.
'''
TrainSample = Train.sample(1124373)


TrainFeatures = TrainSample[['start_year', 'start_month', 'start_day',  'start_dayOfWeek',  'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday','haversineDist', 'geoDist', 'start_geohash', 'end_geohash', 'DistanceStartToCityCenter', 'HeadingToCityCenter']]
            
TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'geoDist', 'start_geohash', 'end_geohash']] = TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'geoDist', 'start_geohash', 'end_geohash']].astype(object)


#baseline model: regression only use haversineDist  numeric features
X_train, X_test, Y_train, Y_test= train_test_split(TrainFeatures, TrainSample['duration'], test_size=0.4, random_state=2017)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
#674623 449750 674623 449750

lm = linear_model.LinearRegression()
lm.fit(X_train[['haversineDist']], Y_train)
predict_train = lm.predict(X_train[['haversineDist']])
predict_test = lm.predict(X_test[['haversineDist']])
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 71446.19
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 71278.77

#for plot of any model
plt.scatter(Y_test, predict_test)
plt.xlabel('duration actual')
plt.ylabel('duration predicted')
plt.title('duration actual vs predicted')
plt.show()

#residual plot
plt.scatter(predict_train, predict_train - Y_train, c = 'b', s = 40, alpha = 0.5)
plt.scatter(predict_test, predict_test - Y_test, c = 'g', s =40)
plt.hlines(y = 0, xmin = 0, xmax = 50)
plt.title('Residual Plot using Train(blue) and Test(green)')
plt.ylabel('residual')

#2nd model: regression use numeric features
X_train_scale=scale(X_train[['haversineDist',  'DistanceStartToCityCenter', 'HeadingToCityCenter']])
X_test_scale=scale(X_test[['haversineDist', 'DistanceStartToCityCenter', 'HeadingToCityCenter']])

lm = linear_model.LinearRegression()
lm.fit(X_train_scale, Y_train)
predict_train = lm.predict(X_train_scale)
predict_test = lm.predict(X_test_scale)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train  70336.25
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test:  70189.63


#3rd model, include categorical features with LabelEncoder
le=LabelEncoder()
#Cat_Train = X_train.select_dtypes(include=[object])
Cat_Train = X_train[['start_year', 'start_month', 'start_dayOfWeek', 'start_hour',  'start_isWeekend',  'start_isHoliday', 'geoDist']]
Cat_Train = Cat_Train.apply(le.fit_transform)
#Cat_Train.head()
enc = OneHotEncoder()
enc.fit(Cat_Train)
onehotlabels_train = enc.transform(Cat_Train).toarray()
#onehotlabels_train.shape


le=LabelEncoder()
Cat_test = X_test[['start_year', 'start_month', 'start_dayOfWeek', 'start_hour',  'start_isWeekend',  'start_isHoliday', 'geoDist']]
Cat_test = Cat_test.apply(le.fit_transform)
#Cat_test.head()
enc = OneHotEncoder()
enc.fit(Cat_test)
onehotlabels_test = enc.transform(Cat_test).toarray()
#onehotlabels_test.shape

lm = linear_model.LinearRegression()
lm.fit(onehotlabels_train, Y_train)
predict_train = lm.predict(onehotlabels_train)
predict_test = lm.predict(onehotlabels_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 102783.13
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 102710.50


#4th model, combine numeric and labelencoded cat features
combined_train = pd.concat([pd.DataFrame(X_train_scale), pd.DataFrame(onehotlabels_train)], axis=1)
combined_test = pd.concat([pd.DataFrame(X_test_scale), pd.DataFrame(onehotlabels_test)], axis=1)

#haversinereset = X_train[['haversineDist']].reset_index(drop=True)
#combined_train = pd.concat([haversinereset, pd.DataFrame(onehotlabels_train)], axis=1, join_axes=[haversinereset.index])
#haversinereset_test = X_test[['haversineDist']].reset_index(drop=True)
#combined_test = pd.concat([haversinereset_test, pd.DataFrame(onehotlabels_test)], axis=1, join_axes=[haversinereset_test.index])

lm = linear_model.LinearRegression()
lm.fit(combined_train, Y_train)
predict_train = lm.predict(combined_train)
predict_test = lm.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 62131.18
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 62030.55

#5th model, regression tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=0)
tree.fit(combined_train, Y_train)
predict_train = tree.predict(combined_train)
predict_test = tree.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 10.68
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 80741.43


#6th model, random forest
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(max_depth=3, random_state=0)
RF.fit(combined_train, Y_train)
predict_train = RF.predict(combined_train)
predict_test = RF.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train  59873.37
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 59754.02


#7th model, GBM
from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GBM = ensemble.GradientBoostingRegressor(**params)
GBM.fit(combined_train, Y_train)
predict_train = GBM.predict(combined_train)
predict_test = GBM.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 51487.63
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 51714.15

# #############################################################################
# Plot feature importance
feature_importance = GBM.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, combined_train.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


#8th model, xgboost
from xgboost import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV

params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)
params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}


XGB = XGBRegressor(nthread=-1)
grid = RandomizedSearchCV(XGB, params, n_jobs=1)
grid.fit(combined_train, Y_train)
predict_train = grid.best_estimator_.predict(combined_train)
predict_test = grid.best_estimator_.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 



