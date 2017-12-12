# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:07:56 2017

@author: nadiao
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Train = pd.read_csv(r'C:\Users\nadiao\Documents\Projects\ZY\train.csv')
Test = pd.read_csv(r'C:\Users\nadiao\Documents\Projects\ZY\test.csv')


Train = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train.csv')
Test = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\test.csv')





Train.info()
Train.head()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12905715 entries, 0 to 12905714
Data columns (total 7 columns):
row_id             int64
start_lng          float64
start_lat          float64
end_lng            float64
end_lat            float64
start_timestamp    int64
duration           int64
dtypes: float64(4), int64(3)
memory usage: 689.2 MB
Out[4]: 
   row_id  start_lng  start_lat    end_lng    end_lat  start_timestamp  \
0       0 -74.009087  40.713818 -74.004326  40.719986       1420950819   
1       1 -73.971176  40.762428 -74.004181  40.742653       1420950819   
2       2 -73.994957  40.745079 -73.999939  40.734650       1421377541   
3       3 -73.991127  40.750080 -73.988609  40.734890       1421377542   
4       4 -73.945511  40.773724 -73.987434  40.755707       1422173586   

   duration  
0       112  
1      1159  
2       281  
3       636  
4       705  
'''

Test.info()
Test.head()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1434344 entries, 0 to 1434343
Data columns (total 6 columns):
row_id             1434344 non-null int64
start_lng          1434344 non-null float64
start_lat          1434344 non-null float64
end_lng            1434344 non-null float64
end_lat            1434344 non-null float64
start_timestamp    1434344 non-null int64
dtypes: float64(4), int64(2)
memory usage: 65.7 MB
Out[5]: 
   row_id  start_lng  start_lat    end_lng    end_lat  start_timestamp
0       0 -73.993111  40.724289 -74.000977  40.735222       1422173589
1       1 -73.971924  40.762749 -73.965698  40.771427       1420567340
2       2 -73.953247  40.765816 -73.952843  40.772453       1420567343
3       3 -73.986618  40.739353 -73.949158  40.805161       1420103336
4       4 -73.968864  40.757317 -73.982521  40.771305       1420690180
'''

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
#count      366.000000
#mean     35261.516393
#std       5113.010975
#min       6728.000000
#25%      32213.000000
#50%      35488.000000
#75%      38775.750000
#max      45666.000000

TimeSeriesTrain.loc[TimeSeriesTrain['Count'] == 6728]['start_date']  # 2015-01-27
TimeSeriesTrain.loc[TimeSeriesTrain['Count'] == 45666]['start_date']  # 2015-01-31

Train['start_year'].value_counts()/Train['start_year'].value_counts().sum()
#2015    0.998889
#2016    0.001111

#2015    12891378
#2016       14337

Train['start_month'].value_counts()/Train['start_month'].value_counts().sum()
#3     0.091410
#5     0.090644
#4     0.089387
#1     0.086480
#2     0.084926
#10    0.084388
#6     0.084249
#7     0.079273
#12    0.078379
#11    0.077472
#9     0.076769
#8     0.076624


Train['start_dayOfWeek'].value_counts()/Train['start_dayOfWeek'].value_counts().sum()
#5    0.155736 
#6    0.151436
#4    0.150238
#3    0.147581
#2    0.142000
#1    0.131846
#0    0.121163

Train['start_isWeekend'].value_counts()/Train['start_isWeekend'].value_counts().sum()
#0.0    0.692828
#1.0    0.307172

Train['start_hour'].value_counts()/Train['start_hour'].value_counts().sum()
#2     0.061415
#3     0.059703
#4     0.058329
#5     0.056617
#1     0.056448
#6     0.051506
#21    0.050242
#22    0.049105
#20    0.049088
#19    0.048637
#0     0.047472
#18    0.046494
#17    0.045643
#16    0.045560
#23    0.044554
#15    0.042333
#7     0.041763
#14    0.031857
#8     0.031436
#9     0.023139
#13    0.018348
#10    0.017018
#11    0.012530
#12    0.010763




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
#count     366.000000
#mean     3918.972678
#std       572.419814
#min       694.000000
#25%      3589.750000
#50%      3946.500000
#75%      4301.000000
#max      5086.000000

TimeSeriesTest.loc[TimeSeriesTest['Count'] == 694]['start_date']  # 2015-01-27
TimeSeriesTest.loc[TimeSeriesTest['Count'] == 5086]['start_date']  # 2015-05-03

Test['start_year'].value_counts()/Test['start_year'].value_counts().sum()
#2015    0.998955
#2016    0.001045

#2015    1432845
#2016       1499

Test['start_month'].value_counts()/Test['start_month'].value_counts().sum()
#3     0.091344
#5     0.090709
#4     0.089476
#1     0.086617
#10    0.084760
#2     0.084666
#6     0.084267
#7     0.079289
#12    0.078595
#11    0.077051
#8     0.076677
#9     0.076549


Test['start_dayOfWeek'].value_counts()/Test['start_dayOfWeek'].value_counts().sum()
#5    0.156032
#6    0.151175
#4    0.150071
#3    0.147518
#2    0.141711
#1    0.132152
#0    0.121340

Test['start_isWeekend'].value_counts()/Test['start_isWeekend'].value_counts().sum()
#0.0    0.692793
#1.0    0.307207

Test['start_hour'].value_counts()/Test['start_hour'].value_counts().sum()
#2     0.061314
#3     0.059696
#4     0.058216
#1     0.056513
#5     0.056076
#6     0.051584
#21    0.050239
#20    0.049181
#22    0.049064
#19    0.048575
#0     0.047367
#18    0.046686
#16    0.045752
#17    0.045650
#23    0.044866
#15    0.042133
#7     0.041687
#14    0.032018
#8     0.031370
#9     0.023239
#13    0.018350
#10    0.017138
#11    0.012602
#12    0.010685

'''
Duration description
'''

Train.duration.describe()
#count    1.290572e+07
#mean     8.421420e+02
#std      7.127489e+02
#min      1.000000e+00
#25%      4.000000e+02
#50%      6.650000e+02
#75%      1.076000e+03
#max      4.317800e+04

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
#count    1.290572e+07
#mean     2.146068e+00
#std      2.418317e+00
#min      0.000000e+00
#25%      7.781379e-01
#50%      1.326174e+00
#75%      2.450537e+00
#max      4.263681e+01

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
#count    1.434344e+06
#mean     2.142921e+00
#std      2.413553e+00
#min      0.000000e+00
#25%      7.777923e-01
#50%      1.325574e+00
#75%      2.449825e+00
#max      3.706941e+01

'''
Location: 
    - Categorized Long/Lat
    - IsBusyLocation(start/end), 

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
#19545.00     5556766
#3803.00      4949163
#625441.00    1674679
#123264.00     357034
#610.00        257686
#0.60           60469
#118.00         26897
#19.00          14425
#3.71            8596


pgh.encode(42.6, -5.6)
# >>> 'ezs42e44yx96'
    
pgh.encode(42.6, -5.6, precision=5)
# >>> 'ezs42'
    
pgh.decode('ezs42')
# >>> (42.6, -5.6)
    
pgh.geohash_approximate_distance('shi3u', 'sh83n')
# >>> 625441


#Train.to_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv',index=False)
#Train.to_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv',index=False)


Train = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv')
#Test = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\test_clean.csv')

Train.head()
Train.geoDist.value_counts()
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
Train.info()
Train.head()
Train.isnull().values.any()

'''
Numeric features:
duration           int64
start_year         int64
start_month        int64
start_day          int64
start_date         object
start_dayOfWeek    int64
start_time         object
start_hour         int64
start_minute       int64
start_second       int64
start_isWeekend    float64
start_isHoliday    int32
haversineDist      float64
geoDist            float64
start_geohash      object
end_geohash        object

'''

import sklearn
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
 
#some correlation plots
plt.scatter(Train['haversineDist'], Train['duration'])
plt.xlabel('haversineDist')
plt.ylabel('duration')
plt.title('haversineDist vs duration')
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

#try data with outliers removed 
removedExtremeDurations = Train.loc[Train['duration'] < 1500]
len(removedExtremeDurations)/len(Train) #0.8822549380046639
plt.hist(removedExtremeDurations.duration, normed=True, bins=50)

SampledRemovedExtremeDurations = removedExtremeDurations.sample(822387)


TrainFeatures = SampledRemovedExtremeDurations[['start_year', 'start_month', 'start_day',  'start_dayOfWeek',  'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday','haversineDist', 'geoDist', 'start_geohash', 'end_geohash']]
            
TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'geoDist', 'start_geohash', 'end_geohash']] = TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'geoDist', 'start_geohash', 'end_geohash']].astype(object)


#baseline model: regression only use numeric features
X_train, X_test, Y_train, Y_test= train_test_split(TrainFeatures, SampledRemovedExtremeDurations['duration'], test_size=0.4, random_state=2017)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
#493432 328955 493432 328955

lm = linear_model.LinearRegression()
lm.fit(X_train[['haversineDist']], Y_train)
predict_train = lm.predict(X_train[['haversineDist']])
predict_test = lm.predict(X_test[['haversineDist']])
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 69972.70
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 70340.33

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

#3rd model, scale numeric features for regression
X_train_scale=scale(X_train[['haversineDist']])
X_test_scale=scale(X_test[['haversineDist']])

lm = linear_model.LinearRegression()
lm.fit(X_train_scale, Y_train)
predict_train = lm.predict(X_train_scale)
predict_test = lm.predict(X_test_scale)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 69972.70
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 70341.41

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

#4th model, include categorical features with LabelEncoder
le=LabelEncoder()
#Cat_Train = X_train.select_dtypes(include=[object])
Cat_Train = X_train[['start_year', 'start_month', 'start_dayOfWeek', 'start_hour',  'start_isWeekend',  'start_isHoliday', 'geoDist']]
Cat_Train = Cat_Train.apply(le.fit_transform)
Cat_Train.head()
enc = OneHotEncoder()
enc.fit(Cat_Train)
onehotlabels_train = enc.transform(Cat_Train).toarray()
onehotlabels_train.shape


le=LabelEncoder()
Cat_test = X_test[['start_year', 'start_month', 'start_dayOfWeek', 'start_hour',  'start_isWeekend',  'start_isHoliday', 'geoDist']]
Cat_test = Cat_test.apply(le.fit_transform)
#Cat_test.head()
enc = OneHotEncoder()
enc.fit(Cat_test)
onehotlabels_test = enc.transform(Cat_test).toarray()
onehotlabels_test.shape

lm = linear_model.LinearRegression()
lm.fit(onehotlabels_train, Y_train)
predict_train = lm.predict(onehotlabels_train)
predict_test = lm.predict(onehotlabels_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 101063.62
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 101564.76
len(lm.coef_)  #54


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

#5th model, combine numeric and labelencoded cat features
#combined_train = pd.concat([X_train[['haversineDist']], pd.DataFrame(onehotlabels_train)], axis=1, join_axes=[X_train.index])
#combined_test = pd.concat([X_test[['haversineDist']], pd.DataFrame(onehotlabels_test)], axis=1, join_axes=[X_test.index])

haversinereset = X_train[['haversineDist']].reset_index(drop=True)
combined_train = pd.concat([haversinereset, pd.DataFrame(onehotlabels_train)], axis=1, join_axes=[haversinereset.index])
haversinereset_test = X_test[['haversineDist']].reset_index(drop=True)
combined_test = pd.concat([haversinereset_test, pd.DataFrame(onehotlabels_test)], axis=1, join_axes=[haversinereset_test.index])

lm = linear_model.LinearRegression()
lm.fit(combined_train, Y_train)
predict_train = lm.predict(combined_train)
predict_test = lm.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 61093.97
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 61509.03

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

#6th model, haversine, hour, dayofweek, not work for one hot encoded data
selected_train = combined_train[['haversineDist', 'start_dayOfWeek', 'start_hour']]
selected_test = combined_test[['haversineDist', 'start_dayOfWeek', 'start_hour']]

lm = linear_model.LinearRegression()
lm.fit(selected_train, Y_train)
predict_train = lm.predict(selected_train)
predict_test = lm.predict(selected_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 69624.45
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 69546.45

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

#7th model, regression tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=0)
tree.fit(combined_train, Y_train)
predict_train = tree.predict(combined_train)
predict_test = tree.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 337.00
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 93301.56

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

#8th model, random forest
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(max_depth=3, random_state=0)
RF.fit(combined_train, Y_train)
predict_train = RF.predict(combined_train)
predict_test = RF.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 58173.49
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 58288.05

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

#9th model, SVR with combined features
from sklearn import svm
svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(combined_train, Y_train)
predict_train = svr_rbf.predict(combined_train)
predict_test = svr_rbf.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 246702.90
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 244700.53

#10th model, kernel ridge with combined features
from sklearn.kernel_ridge import KernelRidge
KR = KernelRidge(alpha=1.0)
KR.fit(combined_train, Y_train)
predict_train = KR.predict(combined_train)
predict_test = KR.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 246702.90
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 244700.53

#haversine transformation, nonlinear


#...th model, use backward difference encoding for cat features:
import category_encoders as ce
BackwardDiff_train = X_train.select_dtypes(include=['object']).copy()
encoder = ce.backward_difference.BackwardDifferenceEncoder()
encoder.fit(BackwardDiff_train, verbose=1)
BackwardDiff_train = encoder.transform(BackwardDiff_train)
BackwardDiff_train.head()




le=LabelEncoder()
Cat_Train = X_train.select_dtypes(include=[object])
Cat_Train = Cat_Train.apply(le.fit_transform)
Cat_Train.head()

le=LabelEncoder()
Cat_test = X_test.select_dtypes(include=[object])
Cat_test = Cat_test.apply(le.fit_transform)
Cat_test.head()

lm = linear_model.LinearRegression()
lm.fit(Cat_Train, Y_train)
predict_train = lm.predict(Cat_Train)
predict_test = lm.predict(Cat_test)
print('Coefficients: \n', lm.coef_) 





