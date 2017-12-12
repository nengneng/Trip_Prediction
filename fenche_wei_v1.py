<<<<<<< HEAD:fenche_v1.py
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

TrainFeatures = removedExtremeDurations[['start_year', 'start_month', 'start_day',  'start_dayOfWeek',  'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday','haversineDist', 'geoDist', 'start_geohash', 'end_geohash']]
            
TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'geoDist', 'start_geohash', 'end_geohash']] = TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'geoDist', 'start_geohash', 'end_geohash']].astype(object)


#baseline model: regression only use numeric features
X_train, X_test, Y_train, Y_test= train_test_split(TrainFeatures, removedExtremeDurations['duration'], test_size=0.4, random_state=2017)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
#4934325 3289550 4934325 3289550

lm = linear_model.LinearRegression()
lm.fit(X_train[['haversineDist']], Y_train)
predict_train = lm.predict(X_train[['haversineDist']])
predict_test = lm.predict(X_test[['haversineDist']])
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 70151.65
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 70083.86

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
#Mean squared error for train 70151.65
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 70083.89

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
Cat_Train = X_train.select_dtypes(include=[object])
Cat_Train = Cat_Train.apply(le.fit_transform)
#Cat_Train.head()

le=LabelEncoder()
Cat_test = X_test.select_dtypes(include=[object])
Cat_test = Cat_test.apply(le.fit_transform)
#Cat_test.head()

lm = linear_model.LinearRegression()
lm.fit(Cat_Train, Y_train)
predict_train = lm.predict(Cat_Train)
predict_test = lm.predict(Cat_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 109110.30
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 110025.56

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
combined_train = pd.concat([X_train[['haversineDist']], Cat_Train], axis=1, join_axes=[X_train.index])
combined_test = pd.concat([X_test[['haversineDist']], Cat_test], axis=1, join_axes=[X_test.index])

lm = linear_model.LinearRegression()
lm.fit(combined_train, Y_train)
predict_train = lm.predict(combined_train)
predict_test = lm.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 67735.14
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 68306.26

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

#6th model, haversine, hour, dayofweek
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
#Mean squared error for train 0.00
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 102015.59

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
#Mean squared error for train 57608.14
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 57588.18

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





=======
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

removedExtremeDurations = Train.loc[Train['duration'] < 1000]
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

#did not use features: start_date, start_time

#transform dytpes:
TrainFeatures = Train[['start_year', 'start_month', 'start_day',  'start_dayOfWeek',  'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday','haversineDist', 'geoDist', 'start_geohash', 'end_geohash']]
            
TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'start_geohash', 'end_geohash']] = TrainFeatures[['start_year', 'start_month', 'start_day', 'start_dayOfWeek', 'start_hour', 'start_minute', 'start_second', 'start_isWeekend',  'start_isHoliday', 'start_geohash', 'end_geohash']].astype(object)

'''
First round, duration as target.
'''
#baseline model: regression only use numeric features
X_train, X_test, Y_train, Y_test= train_test_split(TrainFeatures, Train['duration'], test_size=0.3, random_state=2017)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
#6524998 2796429 6524998 2796429

lm = linear_model.LinearRegression()
lm.fit(X_train[['haversineDist', 'geoDist']], Y_train)
predict_train = lm.predict(X_train[['haversineDist', 'geoDist']])
predict_test = lm.predict(X_test[['haversineDist', 'geoDist']])
print('Coefficients: \n', lm.coef_) 
#Coefficients: 
#   [  2.08374371e+02  -1.55801116e-04]
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error 252147.64
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error: 248125.32

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

#2nd model, using haversineDist only.
lm = linear_model.LinearRegression()
lm.fit(X_train[['haversineDist']], Y_train)
predict_train = lm.predict(X_train[['haversineDist']])
predict_test = lm.predict(X_test[['haversineDist']])
print('Coefficients: \n', lm.coef_) 
#Coefficients: 
#   [ 205.46542746]
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 253120.89
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 249066.52

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
X_train_scale=scale(X_train[['haversineDist', 'geoDist']])
X_test_scale=scale(X_test[['haversineDist', 'geoDist']])

lm = linear_model.LinearRegression()
lm.fit(X_train_scale, Y_train)
predict_train = lm.predict(X_train_scale)
predict_test = lm.predict(X_test_scale)
print('Coefficients: \n', lm.coef_) 
#Coefficients: 
#   [ 500.11243613  -31.96860494]
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 252147.64
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 248125.49

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
#Coefficients: 
# [  0.00000000e+00   1.02581980e+01   6.16759392e-01 ...,  -1.16925906e+02
#   3.33917195e-05  -1.85970465e-05]
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 490352.49
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 488540.30

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
combined_train = pd.concat([X_train[['haversineDist', 'geoDist']], Cat_Train], axis=1, join_axes=[X_train.index])
combined_test = pd.concat([X_test[['haversineDist', 'geoDist']], Cat_test], axis=1, join_axes=[X_test.index])

lm = linear_model.LinearRegression()
lm.fit(combined_train, Y_train)
predict_train = lm.predict(combined_train)
predict_test = lm.predict(combined_test)
print('Coefficients: \n', lm.coef_) 
#Coefficients: 
# [  2.08543323e+02  -7.32428913e-05  -1.98193963e-10 ...,  -1.53142883e+02
#  -1.19691748e-06  -2.31358165e-05]
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 246702.90
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 244700.53

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

#6th model, SVM with combined features
from sklearn import svm
clf = svm.SVR()
clf.fit(combined_train, Y_train)
predict_train = clf.predict(combined_train)
predict_test = clf.predict(combined_test)
print('Coefficients: \n', clf.coef_) 
#Coefficients: 
# [  2.08543323e+02  -7.32428913e-05  -1.98193963e-10 ...,  -1.53142883e+02
#  -1.19691748e-06  -2.31358165e-05]
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 246702.90
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 244700.53





#6h model, use backward difference encoding for cat features:
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







#7th model, include categorical features with LabelEncoder
le=LabelEncoder()
Cat_Train = X_train.select_dtypes(include=[object])
Cat_Train = Cat_Train.apply(le.fit_transform)
Cat_Train.head()

enc = OneHotEncoder()
enc.fit(Cat_Train)

onehotlabels_Train = enc.fit_transform((Cat_Train).as_matrix())
#onehotlabels = enc.transform(Cat_Train).toarray()
#onehotlabels.shape

le=LabelEncoder()
Cat_test = X_test.select_dtypes(include=[object])
Cat_test = Cat_test.apply(le.fit_transform)
Cat_test.head()

enc = OneHotEncoder()
enc.fit(Cat_test)

onehotlabels_test = enc.fit_transform((Cat_test).as_matrix())


regr = linear_model.LinearRegression()
regr.fit(Cat_Train, Y_train)
duration_y_pred = regr.predict(Cat_test)
print('Coefficients: \n', regr.coef_) 
#Coefficients: 
#  [  0.00000000e+00   1.04798014e+01   6.60495089e-01   1.52321316e+01
#   4.65772139e+00  -1.16021115e+02  -1.16269874e+02   1.06234950e-04
#  -6.09924613e-05]
print("Mean squared error: %.2f" % mean_squared_error(Y_test, duration_y_pred))
#Mean squared error: 511824.50
print('Variance score: %.2f' % r2_score(Y_test, duration_y_pred))
#Variance score: -0.04



from sklearn import tree
RF = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
RF.fit(X_train_scale,Y_train)
#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best')
accuracy_score(Y_test,RF.predict(X_test_scale))
#0.65559400607227969


##One hot encoder
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(sparse=False)
X_train_1=X_train
X_test_1=X_test
columns=['DeviceFamily', 'OSSkuName', 'FlightRing', 'OSArchitecture','PrimaryDiskTypeName', 'MDC2FormFactor', 'DisplayLanguage',  'Region','DeviceType', 'AgeGroup', 'Gender', 'BuildNumber', 'Branch']
for col in columns:
       # creating an exhaustive list of all possible categorical values
       data=X_train[[col]].append(X_test[[col]])
       enc.fit(data)
       # Fitting One Hot Encoding on train data
       temp = enc.transform(X_train[[col]])
       # Changing the encoded features into a data frame with new column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # In side by side concatenation index values should be same
       # Setting the index values similar to the X_train data frame
       temp=temp.set_index(X_train.index.values)
       # adding the new One Hot Encoded varibales to the train data frame
       X_train_1=pd.concat([X_train_1,temp],axis=1)
       # fitting One Hot Encoding on test data
       temp = enc.transform(X_test[[col]])
       # changing it into data frame and adding column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # Setting the index for proper concatenation
       temp=temp.set_index(X_test.index.values)
       # adding the new One Hot Encoded varibales to test data frame
       X_test_1=pd.concat([X_test_1,temp],axis=1)

# Standardizing the data set
X_train_scale=scale(X_train_1)
X_test_scale=scale(X_test_1)
# Fitting a logistic regression model
log=LogisticRegression(penalty='l2',C=1)
log.fit(X_train_scale,Y_train)
#LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#          verbose=0, warm_start=False)
# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))
#0.70990173353791908

scores = sklearn.metrics.classification_report(Y_test, log.predict(X_test_scale))
print(scores)
#             precision    recall  f1-score   support
#
#        0.0       0.68      0.79      0.73     30565
#        1.0       0.75      0.63      0.68     30697
#
#avg / total       0.71      0.71      0.71     61262

#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(linewidth=75)

print(log.coef_)

print(log.intercept_)
#[ 0.08490617]


   

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train_scale,Y_train)
#RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#           max_features='auto', max_leaf_nodes=None,
#           min_impurity_split=1e-07, min_samples_leaf=1,
#           min_samples_split=2, min_weight_fraction_leaf=0.0,
#           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
#           verbose=0, warm_start=False)
names = ['VisibleNotificationCount', 'DeviceFamily', 'OSSkuName', 'FlightRing', 'OSArchitecture', 'TotalPhysicalRAM', 'PrimaryDiskTypeName', 'ProcessorPhysicalCores', 'ProcessorCores', 'MDC2FormFactor', 'DisplayLanguage',  'Region','DeviceType', 'AgeGroup', 'Gender', 'BuildNumber', 'Branch', 'RevisionNumber', 'EngagementTimeS']
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))
'''
[(0.2485, 'EngagementTimeS'), (0.067199999999999996, 'TotalPhysicalRAM'), (0.050200000000000002, 'VisibleNotificationCount'), (0.027300000000000001, 'RevisionNumber'), (0.025999999999999999, 'DisplayLanguage'), (0.025000000000000001, 'Region'), (0.021299999999999999, 'ProcessorCores'), (0.0212, 'AgeGroup'), (0.016400000000000001, 'ProcessorPhysicalCores'), (0.014500000000000001, 'OSSkuName'), (0.011299999999999999, 'BuildNumber'), (0.0111, 'FlightRing'), (0.0091999999999999998, 'Gender'), (0.0089999999999999993, 'MDC2FormFactor'), (0.0038, 'DeviceType'), (0.0030999999999999999, 'PrimaryDiskTypeName'), (0.0019, 'OSArchitecture'), (0.001, 'Branch'), (0.0, 'DeviceFamily')]
'''


from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model, 5)
rfe = rfe.fit(X_train_scale,Y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)



from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train_scale,Y_train)
print(model.feature_importances_)


print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 
             reverse=True))
'''
[(0.26900000000000002, 'EngagementTimeS'), (0.091399999999999995, 'VisibleNotificationCount'), (0.064299999999999996, 'TotalPhysicalRAM'), (0.041700000000000001, 'RevisionNumber'), (0.028299999999999999, 'DeviceType'), (0.019900000000000001, 'ProcessorPhysicalCores'), (0.014200000000000001, 'ProcessorCores'), (0.011599999999999999, 'AgeGroup'), (0.010999999999999999, 'DisplayLanguage'), (0.010200000000000001, 'Region'), (0.0092999999999999992, 'BuildNumber'), (0.0074999999999999997, 'OSSkuName'), (0.0066, 'PrimaryDiskTypeName'), (0.0057000000000000002, 'MDC2FormFactor'), (0.0055999999999999999, 'Gender'), (0.0051999999999999998, 'OSArchitecture'), (0.0045999999999999999, 'FlightRing'), (0.0011000000000000001, 'Branch'), (0.00020000000000000001, 'DeviceFamily')]
'''
#np.set_printoptions()














'''
First round, use PassOrFail as target.
'''
X_train, X_test, Y_train, Y_test= train_test_split(DatFeatures, DatAll_downAll_deduped['Passing'], test_size=0.2, random_state=2017)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
#176288 44072 176288 44072


# Standardizing the train and test data
from sklearn.preprocessing import scale
X_train_scale=scale(X_train[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']])
X_test_scale=scale(X_test[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']])


#labelEncode categorical features
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

# Iterating over all the common columns in train and test
for col in X_test.columns.values:
       # Encoding only categorical variables
    if X_test[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
        data=X_train[col].append(X_test[col])
        le.fit(data.values)
        X_train[col]=le.transform(X_train[col])
        X_test[col]=le.transform(X_test[col])


##One hot encoder
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(sparse=False)
X_train_1=X_train
X_test_1=X_test
columns=['DeviceFamily', 'OSSkuName', 'FlightRing', 'OSArchitecture','PrimaryDiskTypeName', 'MDC2FormFactor', 'DisplayLanguage',  'Region','DeviceType', 'AgeGroup', 'Gender', 'BuildNumber', 'Branch']
for col in columns:
       # creating an exhaustive list of all possible categorical values
       data=X_train[[col]].append(X_test[[col]])
       enc.fit(data)
       # Fitting One Hot Encoding on train data
       temp = enc.transform(X_train[[col]])
       # Changing the encoded features into a data frame with new column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # In side by side concatenation index values should be same
       # Setting the index values similar to the X_train data frame
       temp=temp.set_index(X_train.index.values)
       # adding the new One Hot Encoded varibales to the train data frame
       X_train_1=pd.concat([X_train_1,temp],axis=1)
       # fitting One Hot Encoding on test data
       temp = enc.transform(X_test[[col]])
       # changing it into data frame and adding column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # Setting the index for proper concatenation
       temp=temp.set_index(X_test.index.values)
       # adding the new One Hot Encoded varibales to test data frame
       X_test_1=pd.concat([X_test_1,temp],axis=1)

# Standardizing the data set
X_train_scale=scale(X_train_1)
X_test_scale=scale(X_test_1)
# Fitting a logistic regression model
log=LogisticRegression(penalty='l2',C=1)
log.fit(X_train_scale,Y_train)
#LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#          verbose=0, warm_start=False)
accuracy_score(Y_test,log.predict(X_test_scale))
# 0.72365674351061904

scores = sklearn.metrics.classification_report(Y_test, log.predict(X_test_scale))
print(scores)
#             precision    recall  f1-score   support
#
#        0.0       0.69      0.81      0.74     21838
#        1.0       0.77      0.64      0.70     22234
#
#avg / total       0.73      0.72      0.72     44072

#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(linewidth=75)

print(log.coef_)
#[[-0.0643656  -0.07481585  0.03228945 ...,  0.01742683  0.          0.03055134]]

print(log.intercept_)
#[ 0.11294616]


   

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train_scale,Y_train)
#RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#           max_features='auto', max_leaf_nodes=None,
#           min_impurity_split=1e-07, min_samples_leaf=1,
#           min_samples_split=2, min_weight_fraction_leaf=0.0,
#           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
#           verbose=0, warm_start=False)
names = ['VisibleNotificationCount', 'DeviceFamily', 'OSSkuName', 'FlightRing', 'OSArchitecture', 'TotalPhysicalRAM', 'PrimaryDiskTypeName', 'ProcessorPhysicalCores', 'ProcessorCores', 'MDC2FormFactor', 'DisplayLanguage',  'Region','DeviceType', 'AgeGroup', 'Gender', 'BuildNumber', 'Branch', 'RevisionNumber', 'EngagementTimeS']
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))

# make importances relative to max importance
feature_importance = 100.0 * (rf.feature_importances_ / rf.feature_importances_.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

'''
[(0.23760000000000001, 'EngagementTimeS'), (0.053699999999999998, 'TotalPhysicalRAM'), (0.050500000000000003, 'VisibleNotificationCount'), (0.0292, 'FlightRing'), (0.027300000000000001, 'RevisionNumber'), (0.027199999999999998, 'DisplayLanguage'), (0.026599999999999999, 'Region'), (0.020799999999999999, 'AgeGroup'), (0.0167, 'ProcessorPhysicalCores'), (0.0147, 'OSSkuName'), (0.0135, 'ProcessorCores'), (0.0124, 'DeviceType'), (0.0091000000000000004, 'BuildNumber'), (0.0088000000000000005, 'Gender'), (0.0082000000000000007, 'MDC2FormFactor'), (0.0030999999999999999, 'PrimaryDiskTypeName'), (0.0019, 'OSArchitecture'), (0.001, 'Branch'), (0.0, 'DeviceFamily')]

'''

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train_scale,Y_train)
print(model.feature_importances_)
#[  8.19000214e-02   3.56440975e-05   8.32658379e-03 ...,   8.70827070e-07
#   0.00000000e+00   9.04001649e-06]

print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 
             reverse=True))
'''
[(0.25650000000000001, 'EngagementTimeS'), (0.081900000000000001, 'VisibleNotificationCount'), (0.066600000000000006, 'TotalPhysicalRAM'), (0.042799999999999998, 'RevisionNumber'), (0.0247, 'DeviceType'), (0.023900000000000001, 'PrimaryDiskTypeName'), (0.021299999999999999, 'ProcessorPhysicalCores'), (0.0143, 'ProcessorCores'), (0.012, 'DisplayLanguage'), (0.011599999999999999, 'AgeGroup'), (0.010500000000000001, 'Region'), (0.0083000000000000001, 'OSSkuName'), (0.0077999999999999996, 'MDC2FormFactor'), (0.0057000000000000002, 'Gender'), (0.0055999999999999999, 'BuildNumber'), (0.0051000000000000004, 'OSArchitecture'), (0.0037000000000000002, 'FlightRing'), (0.0011000000000000001, 'Branch'), (0.0, 'DeviceFamily')]
'''
#np.set_printoptions()

for name, importance in sorted(zip(model.feature_importances_, names), reverse=True):
    print(name, "=", importance)   
'''
0.254097220876 = EngagementTimeS
0.0817846877791 = VisibleNotificationCount
0.0664166324479 = TotalPhysicalRAM
0.0431640824195 = RevisionNumber
0.0293274216328 = DeviceType
0.0207134940651 = ProcessorPhysicalCores
0.0142610027439 = ProcessorCores
0.0124509316885 = AgeGroup
0.0113656818076 = DisplayLanguage
0.0110250703106 = Region
0.00837632629131 = OSSkuName
0.00730619623485 = PrimaryDiskTypeName
0.00644724746645 = MDC2FormFactor
0.00566745746203 = Gender
0.0051203574513 = BuildNumber
0.00382524601162 = FlightRing
0.00315303031042 = OSArchitecture
0.000808481761945 = Branch
0.000127771233368 = DeviceFamily
'''    



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_scale,Y_train)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
#            verbose=0, warm_start=False)   
for name, importance in sorted(zip(clf.feature_importances_, names), reverse=True):
    print(name, "=", importance)     
'''
0.239317367189 = EngagementTimeS
0.0722053495696 = VisibleNotificationCount
0.0699863771396 = TotalPhysicalRAM
0.0585020065189 = DeviceType
0.035730417737 = RevisionNumber
0.034566190426 = PrimaryDiskTypeName
0.0238730792509 = ProcessorCores
0.0205154655473 = DisplayLanguage
0.0188914852168 = ProcessorPhysicalCores
0.0188259078132 = AgeGroup
0.0182184665692 = Region
0.0153350737671 = MDC2FormFactor
0.0118238117941 = OSSkuName
0.0114093559503 = BuildNumber
0.00771975740702 = Gender
0.0051393411259 = FlightRing
0.00429025923078 = Branch
0.00278516352104 = OSArchitecture
0.000156517865889 = DeviceFamily
'''
#importances = clf.feature_importances_
#indices = np.argsort(importances)
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(DatFeatures, clf.feature_importances_):
    feats[feature] = importance #add the name/value pair 
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)






'''
do not need the following
'''
#check out dist for numeric features
X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int32")]
                        .index.values].hist(figsize=[11,11])

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']],Y_train)
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#           weights='uniform')

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,knn.predict(X_test[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']]))
#0.61511214129476677

LR = LogisticRegression() 
LR.fit(X_train[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']], Y_train)
#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#          verbose=0, warm_start=False)

accuracy_score(Y_test,LR.predict(X_test[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']]))
#0.62764846070973845


# Importing MinMaxScaler and initializing it
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
# Scaling down both train and test data set
X_train_minmax=min_max.fit_transform(X_train[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']])
X_test_minmax=min_max.fit_transform(X_test[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']])

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax, Y_train)
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#           weights='uniform')

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,knn.predict(X_test_minmax))
#0.5877052659070876

LR = LogisticRegression() 
LR.fit(X_train_minmax, Y_train)
#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#          verbose=0, warm_start=False)

accuracy_score(Y_test,LR.predict(X_test[['VisibleNotificationCount', 'TotalPhysicalRAM', 'ProcessorPhysicalCores', 'ProcessorCores', 'EngagementTimeS']]))
#0.50115895661258203



>>>>>>> a8b3ac20676474759d11a0b9bf1fddd57b0ad482:fenche_wei_v1.py
