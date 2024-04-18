#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr  1 22:03:00 2024

DS 2500 Final Project // Temperature’s Influence on Boston Neighborhood Crime Rates 
// Julian Getsey, Jared Mar, and Andrew Chu
"""

# Working with Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning Packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# Machine Learning Metrics
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

def rearrange_dates(df, date_column):
    '''
    Takes in a dataframe and the name of the date_column,
    and rearranges it, converting it into a date_time format
    '''
    for i in df.index:
        date = df.at[i, date_column]
        month = date[:4]
        year = date[4:]
        formatted_date = f"{month}/{year}"
        df.at[i, date_column] = pd.to_datetime(formatted_date)    
    return df

def pick_first_date(df, date_column):
    '''
    Takes in a dataframe and the name of its date column, and returns 
    the dataframe with the first date in the date column in a date_time format'''
    for i in df.index:
        date = str(df.at[i, date_column])
        first_date = date[:11]
        df.at[i, date_column] = pd.to_datetime(first_date)
    return df

def merge_by_date(crime_df, temp_df):
    '''
    Takes in a crime dataframe and a temperature dataframe,
    returning the merged dataframe based on the month
    '''
    crime_df['Month'] = pd.to_datetime(crime_df['Crime Date Time']).dt.month
    crime_df['Year'] = pd.to_datetime(crime_df['Crime Date Time']).dt.year
    
    temp_df['Month'] = pd.to_datetime(temp_df['Date']).dt.month  
    temp_df['Year'] = pd.to_datetime(temp_df['Date']).dt.year
   
    merged_df = pd.merge(crime_df, temp_df, on=['Month', 'Year'])
    
    return merged_df

def add_crimes_per_month(crime_df):
    '''
    Takes in a crime dataframe with month and year columns, and adds a column 
    counting the number of crimes in that row's month and year
    '''
    crimes_per_month = crime_df.groupby(['Year', 'Month']).size()
    crimes_per_month = crimes_per_month.reset_index(name='Crimes per Month')
    merged_df = pd.merge(crime_df, crimes_per_month, on=['Month', 'Year'])
    return merged_df

def find_proportion(list_of_crimes, violent_crime_labels):
    '''
    Takes in a list of crimes and a list of violent crime labels, and returns the proportion of violent crimes
    '''
    violent = 0
    nonviolent = 0
    for crime in list_of_crimes:
        if crime in violent_crime_labels:
            violent += 1
        else:
            nonviolent += 1
    return violent / (violent + nonviolent)

def proportion_of_violent_crimes(crime_df, list_of_violent_crimes, crime_column):
    '''
    Takes in a crime dataframe, a list of violent crimes, and the name of the crime column,
    and creates a new column in the dataframe returning the proportion of violent crimes in the area for that month and year
    '''
    crime_df['Proportion_of_Violent_Crimes'] = 0
    grouped = crime_df.groupby(['Neighborhood', 'Month', 'Year'])
    for name, group in grouped:
        proportion = find_proportion(group[crime_column], list_of_violent_crimes)
        crime_df.loc[group.index, 'Proportion_of_Violent_Crimes'] = proportion
    return crime_df
    
def create_classifier_threshold(crime_df, proportion_column, threshold):
    '''
    Takes in a crime_df, proportion column name, and a decision threshold, and returns 1 or 0 depending
    on whether the proportion is above or below the threshold'''
    crime_df['Proportion_for_Threshold'] = 0
    for i in crime_df.index:
        proportion = (crime_df.at[i, proportion_column])
        if proportion > threshold:
            crime_df.at[i, 'Proportion_for_Threshold'] = 1
        elif proportion <= threshold:
            crime_df.at[i, 'Proportion_for_Threshold'] = 0
    return crime_df
            
def encode_columns(df, cols_to_encode):
    '''
    Takes in a dataframe and columns to encode, and uses a label encoder
    to encode those categorical columns

    '''
    for column in cols_to_encode:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    return df
        
def provide_train_test(df, response_var):
    '''
    Takes in a dataframe and a response variable,
    and returns train and test data by splitting X and Y into features
    and responses
    '''
    X = pd.DataFrame(df.loc[:, df.columns != response_var])
    Y = df[response_var]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    return X_train, X_test, Y_train, Y_test

def mlr_model(X_train, Y_train, X_test, Y_test):
    '''
    Takes in X_train, Y_train, X_test, and a Y_test, creates a 
    Multiple Linear Regression Model, and reports R2 and MSE for train
    and test predictions, returning the model
    '''
    mlr_model = LinearRegression().fit(X_train, Y_train)
    train_predictions = mlr_model.predict(X_train)
    test_predictions = mlr_model.predict(X_test)
    test_r2 = r2_score(Y_test, test_predictions)
    print('MLR: Test R2', test_r2)   
    test_mse = mean_squared_error(Y_test, test_predictions)
    print('MLR: Test MSE', test_mse)
    train_r2 = r2_score(Y_train, train_predictions)
    print('MLRL: Train R2', train_r2)   
    train_mse = mean_squared_error(Y_train, train_predictions)
    print('MLR: Train MSE', train_mse)
    return mlr_model

def plot_coefficients(columns, coefficients):
    '''
    Takes in columns and coefficients, and plots their weights on 
    a bar chart
    '''
    plt.figure(figsize=(16, 4))
    plt.bar(columns, coefficients['Coefficient'], color='blue')
    plt.title('Coefficients/Weights of Each Feature: MLR Model')
    plt.xlabel('Feature')
    plt.ylabel('Assigned Weight')
    plt.show()
    
def find_best_depth(X_train, Y_train, X_test, Y_test):
    '''
    Takes in X_Train, Y_train, X_Test, and Y_test data,
    and returns training and testing errors across a range of max_depths for
    a decision tree
    '''
    maximum_depth_range = range(1, 31)
    training_errors = []
    testing_errors = []
    for depth in maximum_depth_range:
        decision_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=5)
        decision_tree.fit(X_train, Y_train)
        test_predictions = decision_tree.predict(X_test)
        train_predictions = decision_tree.predict(X_train)
        test_error = 1 - accuracy_score(Y_test, test_predictions)
        train_error = 1 - accuracy_score(Y_train, train_predictions)
        testing_errors.append(test_error)
        training_errors.append(train_error)
    return training_errors, testing_errors

def plot_max_depths(training_errors, testing_errors, min_depth, max_depth):
    '''
    Takes in training_errors, testing errors, and a min/max depth, and plots
    the training and testing errors across multiple depths of a decision tree
    '''
    maximum_depth_range = range(min_depth, max_depth+1)
    plt.figure(figsize=(8, 8))
    plt.plot(maximum_depth_range, training_errors, label='Training Error')
    plt.plot(maximum_depth_range, testing_errors, label='Testing Error')
    plt.xlabel('Tree Depth')
    plt.ylabel('Error')
    plt.title('Errors Across Maximum Tree Depths: Decision Tree')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(min(maximum_depth_range)-1, max(maximum_depth_range)+1, step=1))
    plt.show()
    print("Best Maximum Depth:", testing_errors.index(min(testing_errors))+1)
    
def var_importances(X_train, Y_train, X_test, Y_test, decision_tree):
    '''
    Takes in X and Y train, X and Y test data, and returns variable importances of a Decision
    Tree Classifier Model
    '''
    variable_importances = decision_tree.feature_importances_
    variable_importances_dct = {}
    index = 0
    for column_name in X_train.columns:
        variable_importances_dct[column_name] = variable_importances[index]
        index +=1
    return variable_importances_dct

def plot_var_importance(variable_importances_dct):
    '''
    Takes in a variable importances dictionary, and plots these importance 
    values with a bar chart
    '''
    plt.figure(figsize=(10, 4))
    plt.bar(variable_importances_dct.keys(), variable_importances_dct.values(), width = 0.2)
    plt.title('Variable Importance for Each Feature: Decision Tree')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()
    
def plot_confusion_matrix(Y_test, test_predictions):
    '''
    Takes in Y_test data and test predictions, and creates a Confusion Matrix for
    True and False Positive Rates, visualizing a plot of this CM
    '''
    cm = metrics.confusion_matrix(Y_test, test_predictions)
    cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    cm_plot.plot()
    plt.show()
    
def plot_roc_curve(Y_test, test_predictions):
    fpr, tpr, thresholds = roc_curve(Y_test, test_predictions)
    plt.title('ROC Curve for Logistic Regression on Test Data')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0,1], '--')
    plt.plot(fpr, tpr)
    plt.show()


def main():
    # Reading in Crime Report and Weather Data
    crime_report = pd.read_csv("crimereport.csv")
    weather_data = pd.read_csv("WeatherData.csv")
    
    # Removing First 3 Columns from Weather Data (Unnecessary Columns)
    weather_data = weather_data.iloc[3:]
    
    # Creating More Intuitive Names for Columns in Weather Data
    weather_data.rename(columns = {'Boston':'Date'}, inplace = True) 
    weather_data.rename(columns = {' Massachusetts December Average Temperature':'Avg Temp'}, inplace= True)

    # Rearranging Dates / Converting to Date-Time for Weather Data
    weather_data = rearrange_dates(weather_data, 'Date')
    
    # Choosing First Date (Converting to Date-Time) in Range in Crime Report Data
    crime_report = pick_first_date(crime_report, 'Crime Date Time')
   
    # Merging Datasets By Common Month and Year
    crime_and_temp = merge_by_date(crime_report, weather_data)
    
    # Adding Crimes Per Month and Proportion of Violent Crimes
    crime_and_temp = add_crimes_per_month(crime_and_temp)
    crime_and_temp = proportion_of_violent_crimes(crime_and_temp, ['Hit and Run', 'Sex Offender Violation', 'Auto Theft', 'Homicide', 'Street Robbery', 'Aggravated Assault', 'Kidnapping', 'Arson', 'Weapon Violations', 'Commerical Robbery'], 'Crime')

    # Dropping Duplicate Columns in Dataframe to Only Include each Unique Neighborhood/Month/Year Combination
    crime_and_temp = crime_and_temp.drop_duplicates(subset=['Month', 'Year', 'Neighborhood'])
    
    # Removing Unnecessary Columns
    crime_and_temp = crime_and_temp.drop(columns=['File Number', 'Date of Report', 'Crime Date Time', 'Crime', 'Reporting Area', 'Location', 'Date'])
    
    # Encoding Categorical Columns
    encode_columns(crime_and_temp, (['Neighborhood', 'Month', 'Year']))
      
    # Train/Test Data for Linear Regression
    X_train_lin_reg, X_test_lin_reg, Y_train_lin_reg, Y_test_lin_reg = provide_train_test(crime_and_temp, 'Crimes per Month')

    # Performance Metrics for Multiple Linear Regression Model
    m_lin_reg_model= mlr_model(X_train_lin_reg, Y_train_lin_reg, X_test_lin_reg, Y_test_lin_reg)
      
    # Coefficients of Multiple Linear Regression Model
    coefficients = pd.DataFrame(m_lin_reg_model.coef_, X_train_lin_reg.columns, columns=['Coefficient'])
    
    # Plotting Coefficients
    coefficients_plot = plot_coefficients(X_train_lin_reg.columns, coefficients)
    
    # Replacing Proportion of Violent Crimes with A New Column to Determine if it's over a threshold of 20% for classification
    crime_and_temp = create_classifier_threshold(crime_and_temp, 'Proportion_of_Violent_Crimes', 0.2)
    crime_and_temp = crime_and_temp.drop(columns=['Proportion_of_Violent_Crimes'])
    
    # Train/Test Data for Classification Models
    X_train_clf, X_test_clf, Y_train_clf, Y_test_clf = provide_train_test(crime_and_temp, 'Proportion_for_Threshold')
    
    # Finding Training/Testing Errors for a Decision Tree Across Multiple Depths
    training_errors = find_best_depth(X_train_clf, Y_train_clf, X_test_clf, Y_test_clf)[0]
    testing_errors = find_best_depth(X_train_clf, Y_train_clf, X_test_clf, Y_test_clf)[1]

    # Plotting Training and Testing Errors Across Depths
    max_depths_plot = plot_max_depths(training_errors, testing_errors, 1, 30)
    
    # Creating and Fitting Decision Tree with Optimal Max Depth
    decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=5)
    decision_tree.fit(X_train_clf, Y_train_clf)
    
    # Creating Predictions using Decision Tree for Test and Train Data
    dt_test_predictions = decision_tree.predict(X_test_clf)
    dt_train_predictions = decision_tree.predict(X_train_clf)
    
    # False and True Positive Rates, Thresholds for Test and Train Data using predictions
    dt_fpr_test, dt_tpr_test, dt_thresholds_test = metrics.roc_curve(Y_test_clf, dt_test_predictions)
    dt_fpr_train, dt_tpr_train, dt_thresholds_train = metrics.roc_curve(Y_train_clf, dt_train_predictions)
   
    # Find Variable Importances for Decision Tree Model (Max Depth 2)
    variable_importances = var_importances(X_train_clf, Y_train_clf, X_test_clf, Y_test_clf, decision_tree)
    
    # Plotting Variable Importances for DT
    var_importances_plot = plot_var_importance(variable_importances)
    
    # Reporting Performance Metrics for Test Data
    print(f"Decision Tree Information Gain Splitting Criteria / Test Data / Accuracy: {accuracy_score(Y_test_clf, dt_test_predictions)}")
    print(f"Decision Tree Information Gain Splitting Criteria / Test Data / Error: {1 - accuracy_score(Y_test_clf, dt_test_predictions)}")
    print(f"Decision Tree Information Gain Splitting Criteria / Test Data / F1 Score: {f1_score(Y_test_clf, dt_test_predictions)}")
    print(f"Decision Tree Information Gain Splitting Criteria / Test Data / AUC Score: {metrics.auc(dt_fpr_test, dt_tpr_test)}")
        
    # Reporting Performance Metrics for Train Data
    print(f"Decision Tree Information Gain Splitting Criteria / Train Data / Accuracy: {accuracy_score(Y_train_clf, dt_train_predictions)}")
    print(f"Decision Tree Information Gain Splitting Criteria / Train Data / Error: {1 - accuracy_score(Y_train_clf, dt_train_predictions)}")
    print(f"Decision Tree Information Gain Splitting Criteria / Train Data / F1 Score: {f1_score(Y_train_clf, dt_train_predictions)}")
    print(f"Decision Tree Information Gain Splitting Criteria / Train Data / AUC Score: {metrics.auc(dt_fpr_train, dt_tpr_train)}")    

    # Plotting Confusion Matrix for Test Data, Decision Tree
    confusion_matrix = plot_confusion_matrix(Y_test_clf, dt_test_predictions)

    # Creating Logistic Regression Model and Fitting it to the Classification Data
    lr_model = LogisticRegression()
    lr_model.fit(X_train_clf, Y_train_clf)
    
    # Test and Train Predictions for Logistic Regression Model
    lr_test_predictions = lr_model.predict_proba(X_test_clf)[:,1]
    lr_train_predictions = lr_model.predict_proba(X_train_clf)[:,1]
    
    # ROC Curve of Test Prediction TPR and FPR rates for the Logistic Regression Model
    roc_curve = plot_roc_curve(Y_test_clf, lr_test_predictions)

if __name__ == "__main__":
    main()