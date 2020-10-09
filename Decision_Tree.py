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

#sort data according to time
dataset_3['date'] = pd.to_datetime(dataset_3['date'])
dataset_3 = dataset_3.set_index(dataset_3['date'])
dataset_3 = dataset_3.sort_index()

### TASK 3

#convert dataframe into numpy array for calculations
X = dataset_3.iloc[:, 4:-1].values  #get all the columns except the last one
y = dataset_3.iloc[:, -1].values  #get just the last column

# One Hot Encoding the "open_covid_region_code" & "date" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Splitting the dataset into the Training set and Test set according to region (strategy #1)
from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y,train_size = 0.8, test_size = 0.2, random_state = 0, shuffle = True)


#Split the data at index 369 as that will contain all the data till 2020-08-10 for our training set (startagey #2)
X_train_2 = X[:369, :]
y_train_2 = y[:369]

X_test_2 = X[369:, :]
y_test_2 = y[369:]


#Decision Tree

#Training the Decision Tree Regression model on the regions split dataset
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state = 0)
dtree.fit(X_train_1, y_train_1)
dtree.fit(X_train_2, y_train_2)

# prediction
y_pred_1 = dtree.predict(X_test_1)
y_pred_2 = dtree.predict(X_test_2)
df1 = pd.DataFrame({'Actual':y_test_1, 'Predicted':y_pred_1})
df2 = pd.DataFrame({'Actual':y_test_2, 'Predicted':y_pred_2})


#accuracy
from sklearn import metrics

print('')

print('Decision Tree Strategy #1 Mean Absolute Error:', metrics.mean_absolute_error(y_test_1, y_pred_1))
print('Decision Tree Strategy #1 Mean Squared Error:', metrics.mean_squared_error(y_test_1, y_pred_1))
print('Decision Tree Strategy #1 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_1, y_pred_1)))

print('')

print('Decision Tree Strategy #2 Mean Absolute Error:', metrics.mean_absolute_error(y_test_2, y_pred_2))
print('Decision Tree Strategy #2 Mean Squared Error:', metrics.mean_squared_error(y_test_2, y_pred_2))
print('Decision Tree Strategy #2 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_2, y_pred_2)))





























