# Importing the libraries
import numpy as np
import pandas as pd
#import tensorflow as tf



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

pd.set_option('display.max_columns', None)
#print(dataset_2)

print("Dataset 1 DATE")
#Printing DATE of first 4 Rows of dataset 2
print(dataset_2[:4].date.to_string(index=False))

print("Dataset 2 DATE")
#Printing DATE of first 4 Rows of dataset 1
print(dataset_1[:4].date.to_string(index=False))
