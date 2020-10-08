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
dataset_3 = dataset_3.dropna(axis='rows', thresh=0)

### TASK 3

# ## K-NN

#convert dataframe into numpy array for calculations
X = dataset_3.iloc[:, 4:-1].values  #get all the columns except the last one
y = dataset_3.iloc[:, -1].values  #get just the last column

# One Hot Encoding the "open_covid_region_code" & "date" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.8, test_size = 0.2, random_state = 0)

#training the KNN model on the training data
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

#predicting
from sklearn import metrics
y_pred = knn.predict(X_test)
print('predicted   Actual')
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(knn.score(X_test, y_test))

#accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('Decision Tree Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Decision Tree Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Decision Tree Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## Decision Tree

# # Training the Decision Tree Regression model on the whole dataset
# from sklearn.tree import DecisionTreeRegressor
# dtree = DecisionTreeRegressor(random_state = 0)
# dtree.fit(X_train, y_train)

# #prediction
# y_pred = dtree.predict(X_test)
# df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})


# #accuracy
# from sklearn import metrics
# print('Decision Tree Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Decision Tree Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Decision Tree Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


















