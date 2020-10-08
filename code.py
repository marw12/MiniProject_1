# Importing the libraries
import numpy as np
import pandas as pd


### TASK 1

#Data Preprocessing

# Importing the dataset
dataset_1 = pd.read_csv('2020_US_weekly_symptoms_dataset.csv')  #Search Trends dataset
dataset_2 = pd.read_csv('aggregated_cc_by.csv', dtype={"test_units": "object"})  #hospitalization cases dataset
dataset_2 = dataset_2.iloc[78164:90022]  #Loading rows for USA only and rows that don't have all missing values

#Cleaning the datasets

dataset_1 = dataset_1.dropna(axis='columns', how='all')  #removes columns with all NaN values
dataset_1 = dataset_1.dropna(axis='rows', how='all')  #removes rows with all NaN values

dataset_2 = dataset_2.dropna(axis='columns', how='all')  #removes columns with all NaN values
dataset_2 = dataset_2.dropna(axis='rows', how='all')  #removes rows with all NaN values

#Match time resolution

#chnaging date formt to datetime64
dataset_1['date'] = pd.to_datetime(dataset_1['date'])
dataset_2['date'] = pd.to_datetime(dataset_2['date'])
dataset_2 = dataset_2.groupby(['open_covid_region_code', pd.Grouper(key='date', freq='W-MON')])['hospitalized_new'].sum().reset_index().sort_values('date')

#Merge the two datasets

#merging on multiple columns
dataset_3 = pd.merge(dataset_1, dataset_2[['date', 'open_covid_region_code', 'hospitalized_new']], on=['date', "open_covid_region_code"])


### Task 2.1

#First we get the top 5 symptoms 
validCounts = dataset_3.count()
validCounts = validCounts.drop(['open_covid_region_code', 'country_region', 'sub_region_1', 'sub_region_1_code', 'country_region_code', 'date', 'hospitalized_new'])
validCounts = validCounts.sort_values(ascending = False)
validCounts = validCounts.iloc[0:5]
topSymptoms = validCounts.index.tolist()

#Now that we have the top symptoms as a list we can plot them for the visualization