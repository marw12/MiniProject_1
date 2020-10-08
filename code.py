# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

#Store a list of strings the top5 symptoms and collumns we want to keep
noDrop = ['date', 'sub_region_1'] + topSymptoms

#Now that we have the top symptoms as a list we can plot them for the visualization
#We start by taking dataset_3 and dropping all collumns except location,date, symptoms
simplifiedSet = dataset_3[noDrop]

#We also drop any states which have nan values inside all their collumns, we want states with data for the full period we're looking at
statesToDrop = ['Alaska', 'Wyoming', 'Hawaii', 'Idaho', 'North Dakota', 'South Dakota']
for x in range(0, len(statesToDrop)):
    simplifiedSet = simplifiedSet[simplifiedSet.sub_region_1 != statesToDrop[x]]
    
#We now have a list where every state has at least one complete collumn with no Nan values

#Lets plot the search frequency for shallowbreathing first. We will do states DC, Delaware, Maine, West Virginia, RHode Island, New Hampshire
#Weeks will be the same for all so
weekX = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39]

#Now we need to pull out the relevant values for the symptom/weeks we want
shallowDCY = simplifiedSet[simplifiedSet.sub_region_1 == 'District of Columbia']
shallowDCY = shallowDCY[['symptom:Shallow breathing']]
shallowDCY = shallowDCY.values.tolist()

shallowDelawareY = simplifiedSet[simplifiedSet.sub_region_1 == 'Delaware']
shallowDelawareY = shallowDelawareY[['symptom:Shallow breathing']]
shallowDelawareY = shallowDelawareY.values.tolist()

shallowMaineY = simplifiedSet[simplifiedSet.sub_region_1 == 'Maine']
shallowMaineY = shallowMaineY[['symptom:Shallow breathing']]
shallowMaineY = shallowMaineY.values.tolist()

shallowWestVirY = simplifiedSet[simplifiedSet.sub_region_1 == 'West Virginia']
shallowWestVirY = shallowWestVirY[['symptom:Shallow breathing']]
shallowWestVirY = shallowWestVirY.values.tolist()

shallowRhodeY = simplifiedSet[simplifiedSet.sub_region_1 == 'Rhode Island']
shallowRhodeY = shallowRhodeY[['symptom:Shallow breathing']]
shallowRhodeY = shallowRhodeY[1::]
shallowRhodeY = shallowRhodeY.values.tolist()

shallowHampshireY = simplifiedSet[simplifiedSet.sub_region_1 == 'New Hampshire']
shallowHampshireY = shallowHampshireY[['symptom:Shallow breathing']]
shallowHampshireY = shallowHampshireY.values.tolist()

#Now we plot
plt.plot(weekX, shallowDCY, label = 'District of Columbia')
plt.plot(weekX, shallowDelawareY, label = 'Delaware')
plt.plot(weekX, shallowMaineY, label = 'Maine')
plt.plot(weekX, shallowRhodeY, label = 'Rhode Island')
plt.plot(weekX, shallowHampshireY, label = 'New Hampshire')
plt.xlabel('Week')
plt.ylabel('Frequency')
plt.title('Search Frequency for the Symptom: Shallow Breathing')
plt.legend()
plt.show()

#Now lets plot the search frequency for Ventricular fibrillation. We will plot the following states: DC, Delaware, Maine, New Hampshire, RhodeIsland, West Virginia
ventGraph = simplifiedSet[['sub_region_1','symptom:Ventricular fibrillation']]

ventDCY = ventGraph[ventGraph.sub_region_1 == 'District of Columbia']
ventDCY = ventDCY[['symptom:Ventricular fibrillation']]
ventDCY = ventDCY.values.tolist()

ventDelY = ventGraph[ventGraph.sub_region_1 == 'Delaware']
ventDelY= ventDelY[['symptom:Ventricular fibrillation']]
ventDelY = ventDelY.values.tolist()

ventMaineY = ventGraph[ventGraph.sub_region_1 == 'Maine']
ventMaineY = ventMaineY[['symptom:Ventricular fibrillation']]
ventMaineY = ventMaineY.values.tolist()

ventNHY = ventGraph[ventGraph.sub_region_1 == 'New Hampshire']
ventNHY = ventNHY[['symptom:Ventricular fibrillation']]
ventNHY = ventNHY.values.tolist()

ventRIY = ventGraph[ventGraph.sub_region_1 == 'Rhode Island']
ventRIY = ventRIY[['symptom:Ventricular fibrillation']]
ventRIY = ventRIY[1::]
ventRIY = ventRIY.values.tolist()

ventWVY = ventGraph[ventGraph.sub_region_1 == 'West Virginia']
ventWVY = ventWVY[['symptom:Ventricular fibrillation']]
ventWVY = ventWVY.values.tolist()

#Plot the Ventricular Fibrillation graph
plt.plot(weekX, ventDCY, label = 'District of Columbia')
plt.plot(weekX, ventDelY, label = 'Delaware')
plt.plot(weekX, ventMaineY, label = 'Maine')
plt.plot(weekX, ventNHY, label = 'New Hampshire')
plt.plot(weekX, ventRIY, label = 'Rhode Island')
plt.plot(weekX, ventWVY, label = 'West Virginia')
plt.xlabel('Week')
plt.ylabel('Frequency')
plt.title('Search Frequency for the Symptom: Ventricular Fibrillation')
plt.legend()
plt.show()

#Now lets do the same for the symptom aphonia. We will plot the following states: Maine, Nebraska, New Mexico
aphoniaGraph = simplifiedSet[['sub_region_1','symptom:Aphonia']]

aphoniaMaineY = aphoniaGraph[aphoniaGraph.sub_region_1 == 'Maine']
aphoniaMaineY = aphoniaMaineY[['symptom:Aphonia']]
aphoniaMaineY = aphoniaMaineY.values.tolist()

aphoniaNebraskaY = aphoniaGraph[aphoniaGraph.sub_region_1 == 'Nebraska']
aphoniaNebraskaY = aphoniaNebraskaY[['symptom:Aphonia']]
aphoniaNebraskaY = aphoniaNebraskaY.values.tolist()

aphoniaNewMY = aphoniaGraph[aphoniaGraph.sub_region_1 == 'New Mexico']
aphoniaNewMY = aphoniaNewMY[['symptom:Aphonia']]
aphoniaNewMY = aphoniaNewMY.values.tolist()

#Plot now
plt.plot(weekX, aphoniaMaineY, label = 'Maine')
plt.plot(weekX, aphoniaNebraskaY, label = 'Nebraska')
plt.plot(weekX, aphoniaNewMY, label = 'New Mexico')
plt.xlabel('Week')
plt.ylabel('Frequency')
plt.title('Search Frequency for the Symptom: Aphonia')
plt.legend()
plt.show()

