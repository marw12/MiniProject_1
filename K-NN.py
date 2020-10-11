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
dataset_1 = dataset_1.dropna(axis='rows', how='all', thresh=20)  #removes rows with all NaN values

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
dataset_3 = dataset_3.set_index(dataset_3['date'])
dataset_3 = dataset_3.sort_index()

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


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Splitting the dataset into the Training set and Test set according to region (strategy #1)
from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y,train_size = 0.8, test_size = 0.2, random_state = 0, shuffle = True)

#Split the data at index 369 as that will contain all the data till 2020-08-10 for our training set (startagey #2)
# X_train_2 = X[:369, :]
# y_train_2 = y[:369]

# X_test_2 = X[369:, :]
# y_test_2 = y[369:]

#training the KNN model on the training data
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 13, metric = 'minkowski', p = 2)
knn.fit(X_train_1, y_train_1)
# knn.fit(X_train_2, y_train_2)

#predicting
from sklearn import metrics
y_pred_1 = knn.predict(X_test_1)
# y_pred_2 = knn.predict(X_test_2)
print('predicted   Actual')
print(np.concatenate((y_pred_1.reshape(len(y_pred_1),1), y_test_1.reshape(len(y_test_1),1)),1))
print(knn.score(X_test_1, y_test_1))

#accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('KNN Strategy #1 Mean Absolute Error:', metrics.mean_absolute_error(y_test_1, y_pred_1))
print('KNN Strategy #1 Mean Squared Error:', metrics.mean_squared_error(y_test_1, y_pred_1))
print('KNN Strategy #1 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_1, y_pred_1)))

print('')

# print('KNN Strategy #2 Mean Absolute Error:', metrics.mean_absolute_error(y_test_2, y_pred_2))
# print('KNN Strategy #2 Mean Squared Error:', metrics.mean_squared_error(y_test_2, y_pred_2))
# print('KNN Strategy #2 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_2, y_pred_2)))


#search for an optimal value of K for KNN

# range of k we want to try
# k_range = range(1, 31)
# # empty list to store scores
# k_scores = []

# 1. we will loop through reasonable values of k
# for k in k_range:
#     # 2. run KNeighborsClassifier with k neighbours
#     knn = KNeighborsRegressor(n_neighbors=k)
#     # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
#     scores = cross_val_score(knn, X, y, cv=5)
#     # 4. append mean of scores for k neighbors to k_scores list
#     k_scores.append(scores.mean())

# print('Length of list', len(k_scores))
# print('Max of list', max(k_scores))

# # plot how accuracy changes as we vary k
# import matplotlib.pyplot as plt


# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
#plt.plot(x_axis, y_axis)
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-validated accuracy')

# #define a function for the MSE loss
# loss = lambda y, yh: np.mean((y-yh)**2)

# #Plot the mean square error for different K values stored in K_list
# K_list = range(1,31)
# err_train, err_test = [], []
# for i, K in enumerate(K_list):
#     knn = KNeighborsRegressor(n_neighbors=K)
#     knn = knn.fit(X_train, y_train)
#     err_test.append(loss(knn.predict(X_test), y_test))
#     err_train.append(loss(knn.predict(X_train), y_train))

# plt.plot(K_list, err_test, '-', label='unseen')
# plt.plot(K_list, err_train, '-', label='train')
# plt.legend()
# plt.xlabel('K (number of neighbours)')
# plt.ylabel('mean squared error')
# plt.show()




















