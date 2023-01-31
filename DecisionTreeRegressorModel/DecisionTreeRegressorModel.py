#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# # Importing the data

# In[2]:


import pandas as pd
tripdata_df = pd.read_parquet(path = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-02.parquet', #provide the URL to the data source
                      engine = 'fastparquet')


# # Examining the data

# In[3]:


#1 Identifying its dimensions
print('There are {} rows and {} columns.'.format(tripdata_df.shape[0], tripdata_df.shape[1])) 
print(tripdata_df.info())
# Indicating if the variables have suitable types
print(tripdata_df.head())
print(tripdata_df.isnull().sum())
print(tripdata_df.isnull().sum(axis=1).sort_values(ascending = False))


# # Much of the data is missing, so it is a WIDESPREAD issue as opposed to an isolated case. There are at most 7 variables with missing data. Considering there are 20 columns, we could have rows missing +35% of data. We should delete these rows, since they are not very useful to us. We can perform imputation on the rows that have less than 35% missing data for replacement purposes. 

# # Data Preparation

# In[4]:


# Step 1: Imputation: For all missing data that is less than or equal to 35% of the data, we will perform imputation

print(tripdata_df['store_and_fwd_flag'].value_counts(dropna=False))
tripdata_df.loc[tripdata_df['store_and_fwd_flag'].isna(), 'store_and_fwd_flag'] = 'N'

print(tripdata_df['RatecodeID'].value_counts(dropna=False))
tripdata_df.loc[tripdata_df['RatecodeID'].isna(), 'RatecodeID'] = '1.0'

print(tripdata_df['passenger_count'].value_counts(dropna=False))
tripdata_df.loc[tripdata_df['passenger_count'].isna(), 'passenger_count'] = '1.0'

print(tripdata_df['payment_type'].value_counts(dropna=False))
tripdata_df.loc[tripdata_df['payment_type'].isna(), 'payment_type'] = '1.0'

print(tripdata_df['trip_type'].value_counts(dropna=False))
tripdata_df.loc[tripdata_df['trip_type'].isna(), 'trip_type'] = '1.0'

print(tripdata_df['congestion_surcharge'].value_counts(dropna=False))
tripdata_df.loc[tripdata_df['congestion_surcharge'].isna(), 'congestion_surcharge'] = '0.00'


# Step 2: Converting Fields to a Suitable Data Type
tripdata_df['passenger_count'] = tripdata_df['passenger_count'].astype(float)
tripdata_df['RatecodeID'] = tripdata_df['RatecodeID'].astype(float)
tripdata_df['payment_type'] = tripdata_df['payment_type'].astype(float)
tripdata_df['trip_type'] = tripdata_df['trip_type'].astype(float)

#Step 3: Removing any duplicate rows
tripdata_df.drop_duplicates(inplace=True)


# # Partitioning the data into train/test split.

# In[5]:


X = tripdata_df[['passenger_count', 'trip_type', 'trip_distance', 'tolls_amount', 'fare_amount']]
y = tripdata_df['tip_amount']

X_train, X_test, y_train, y_test = train_test_split( 
                                        X, y, test_size = 0.3, random_state = 5
                                        ) 


# In[6]:


# Above, I partitioned the data into X and Y. X is made up of columns that are relevant in predicting the
# variable in our Y, which is tip amount. Therefore, passenger_count, trip_type, etc. could all help us predict
# a tip. 


# # For my features, I selected passenger_count, trip_type, trip_distance, tolls_amount, and fare_amount. These will all help us predict tip_amount. The categorical feature is trip_type. However, I have already encoded trip_type as an integer. 
# 

# # Decision Tree

# In[7]:


dt = DecisionTreeRegressor(random_state=7)
dt.fit(X_train, y_train)


# # Above, I built the decision tree, and chose regression as the data is continuous. 

# # MSE

# In[8]:


# Predicting the labels for the test set
y_pred   = dt.predict(X_test)

print('The predicted tip amount is: {}'.format(y_pred))

mse = mean_squared_error(y_test, y_pred)

# Evaluating the Predictions
print('The mse of the model is: {}'.format(mse))


# # Because the MSE is about 4.6, the model could be improved, however it is still useful/effective for predicting the tip amount. Ideally, we would want an MSE as close to 0 as possible

# In[9]:


# Here I am testing n_estimators=10 for the model to see how the mse is affected
rf = RandomForestRegressor(random_state=7, n_estimators= 10)
rf.fit(X_train, y_train)
y_pred   = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Evaluating the predictions
print('The mse of the model is with n_estimators = 10 is : {}'.format(mse))




# In[10]:


# Here I am testing n_estimators= 20 for the model to see how the mse is affected
rf2 = RandomForestRegressor(random_state=7, n_estimators= 20)
rf2.fit(X_train, y_train)
y_pred   = rf2.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Evaluating the predictions
print('The mse of the model with n_estimators = 20 is: {}'.format(mse))


# In[ ]:


# Here I am testing n_estimators=50 for the model to see how the mse is affected
rf3 = RandomForestRegressor(random_state=7, n_estimators= 50)
rf3.fit(X_train, y_train)
y_pred   = rf3.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Evaluating the predictions
print('The mse of the model with n_estimators = 50 is: {}'.format(mse))


# # I will use this tweaked version of the model. It has the highest n_estimators, and the mse is the closest to 0, so it is the most effective version. 

# # Predictions

# In[ ]:


# Here, I used the feature_importances_, which assigns a float to represent
# the importance of each feature in the random forest. 
rf3.feature_importances_
#'Order of array: passenger_count', 'trip_type', 'trip_distance', 'tolls_amount', 'fare_amount'


# # We see that trip_distance is the most important factor in influencing the tip_amount. After that, fare_amount also plays a large role. However, passenger_count, trip_type, and tolls_amount play a much lesser role in influencing the tip_amount. 
