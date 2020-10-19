# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



### TASK 1

#Data Preprocessing

# Importing the dataset
dataset_1 = pd.read_csv('2020_US_weekly_symptoms_dataset.csv')  #Search Trends dataset

cases_data = pd.read_csv('us-states.csv')
cases_data.rename(columns={"state": "sub_region_1"}, inplace = True)

dataset_2 = pd.read_csv('aggregated_cc_by.csv', dtype={"test_units": "object"})  #hospitalization cases dataset
dataset_2 = dataset_2.iloc[78164:90022]  #Loading rows for USA only and rows that don't have all missing values

daily_data = pd.read_csv('2020_US_daily_symptoms_dataset.csv')

weather_data = pd.read_csv('temperature_dataframe_editUS.csv')
weather_data = weather_data.iloc[12476:15836, 1:]

#Cleaning the datasets

dataset_1 = dataset_1.dropna(axis='columns', how='all', thresh=255)  #removes columns with all NaN values
dataset_1 = dataset_1.dropna(axis='rows', how='all', thresh=20)  #removes rows with all NaN values

dataset_2 = dataset_2.dropna(axis='columns', how='all')  #removes columns with all NaN values
dataset_2 = dataset_2.dropna(axis='rows', how='all')  #removes rows with all NaN values

#Match time resolution

#chnaging date formt to datetime64
dataset_1['date'] = pd.to_datetime(dataset_1['date'])
dataset_2['date'] = pd.to_datetime(dataset_2['date'])
cases_data['date'] = pd.to_datetime(cases_data['date'])
daily_data['date'] = pd.to_datetime(daily_data['date'])
weather_data['date'] = pd.to_datetime(weather_data['date'])


dataset_2 = dataset_2.groupby(['open_covid_region_code', pd.Grouper(key='date', freq='W-MON')])['hospitalized_new'].sum().reset_index().sort_values('date')
cases_data = cases_data.groupby(['sub_region_1', pd.Grouper(key='date', freq='W-MON')])['cases'].sum().reset_index().sort_values('date')
# daily_data = daily_data.groupby(['sub_region_1', pd.Grouper(key='date', freq='W-MON')])[['symptom:Common cold', 'symptom:Shortness of breath', 'symptom:Fever', 'symptom:Cough']].mean().reset_index().sort_values('date')
weather_data = weather_data.groupby(['sub_region_1', pd.Grouper(key='date', freq='W-MON')])['temp'].mean().reset_index().sort_values('date')


#Merge the two datasets

# merging on multiple columns
# dataset_3 = pd.merge(dataset_1, daily_data[['date', 'sub_region_1', 'symptom:Common cold', 'symptom:Fever', 'symptom:Cough', 'symptom:Shortness of breath']], on=['date', 'sub_region_1'])
# dataset_3 = pd.merge(dataset_3, cases_data[['date', 'sub_region_1', 'cases']], on=['date', "sub_region_1"])
dataset_3 = pd.merge(dataset_1, weather_data[['date', 'sub_region_1', 'temp']], on=['date', 'sub_region_1'])
dataset_3 = pd.merge(dataset_3, dataset_2[['date', 'open_covid_region_code', 'hospitalized_new']], on=['date', "open_covid_region_code"])

dataset_3 = dataset_3.set_index(dataset_3['date'])
dataset_3 = dataset_3.sort_index()

## TASK 3

## K-NN

# convert dataframe into numpy array for calculations
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

# Split the data at index 270 as that will contain all the data till 2020-08-10 for our training set (startagey #2)
# X_train_2 = X[:270, :]
# y_train_2 = y[:270]

# X_test_2 = X[270:, :]
# y_test_2 = y[270:]


#training the KNN model on the training data
from sklearn.neighbors import KNeighborsRegressor
knn_1 = KNeighborsRegressor(n_neighbors = 8, metric = 'minkowski', p = 2)
knn_2 = KNeighborsRegressor(n_neighbors = 8, metric = 'minkowski', p = 2)
knn_1.fit(X_train_1, y_train_1)
# knn_2.fit(X_train_2, y_train_2)


#Using Pearson Correlation
plt.figure(figsize=(45,45))
cor = dataset_3.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

#Correlation with output variable
cor_target = abs(cor["hospitalized_new"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
print(relevant_features)

#predicting
from sklearn import metrics
y_pred_1_test = knn_1.predict(X_test_1)
y_pred_1_train = knn_1.predict(X_train_1)
# y_pred_2_test = knn_2.predict(X_test_2)
# y_pred_2_train = knn_2.predict(X_train_2)

print('predicted   Actual')
df1 = pd.DataFrame({'Actual':y_test_1, 'Predicted':y_pred_1_test})
print('score: ', knn_1.score(X_test_1, y_test_1))

#accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_1, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('')                   
print('K value: 8')
print('')

print('Test Data KNN Strategy #1 Mean Absolute Error:', metrics.mean_absolute_error(y_test_1, y_pred_1_test))
print('Test Data KNN Strategy #1 Mean Squared Error:', metrics.mean_squared_error(y_test_1, y_pred_1_test))
print('Test Data KNN Strategy #1 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_1, y_pred_1_test)))

print('')

print('Train Data KNN Strategy #1 Mean Absolute Error:', metrics.mean_absolute_error(y_train_1, y_pred_1_train))
print('Train Data KNN Train Strategy #1 Mean Squared Error:', metrics.mean_squared_error(y_train_1, y_pred_1_train))
print('Train Data KNN Train Strategy #1 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train_1, y_pred_1_train)))

print('')
print('K value: 8')
print('')

# print('Test Data KNN Strategy #2 Mean Absolute Error:', metrics.mean_absolute_error(y_test_2, y_pred_2_test))
# print('Test Data KNN Strategy #2 Mean Squared Error:', metrics.mean_squared_error(y_test_2, y_pred_2_test))
# print('Test Data KNN Strategy #2 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_2, y_pred_2_test)))

# print('')

# print('Train Data KNN Strategy #2 Mean Absolute Error:', metrics.mean_absolute_error(y_train_2, y_pred_2_train))
# print('Train Data KNN Strategy #2 Mean Squared Error:', metrics.mean_squared_error(y_train_2, y_pred_2_train))
# print('Train Data KNN Strategy #2 Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train_2, y_pred_2_train)))


# # search for an optimal value of K for KNN

# range of k we want to try
k_range = range(1, 71)
# empty list to store scores
k_scores = []

# 1. we will loop through reasonable values of k
# for k in k_range:
#     # 2. run KNeighborsClassifier with k neighbours
#     knn = KNeighborsRegressor(n_neighbors=k, metric = 'minkowski', p = 2)
#     # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
#     scores = cross_val_score(knn, X, y, cv=5)
#     # 4. append mean of scores for k neighbors to k_scores list
#     k_scores.append(scores.mean())

# print('Length of list', len(k_scores))
# print('Max of list', max(k_scores))

# # plot how accuracy changes as we vary k


# # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# # plt.plot(x_axis, y_axis)
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-validated accuracy')

#define a function for the MSE loss
loss = lambda y, yh: np.mean((y-yh)**2)

# Plot the mean square error for different K values stored in K_list
# K_list = range(1,51)
# err_train_2, err_test_2 = [], []
# for i, K in enumerate(K_list):
#     knn_2 = KNeighborsRegressor(n_neighbors=K, metric = 'minkowski', p = 2)
#     knn_2 = knn_2.fit(X_train_2, y_train_2)
#     err_train_2.append(loss(knn_2.predict(X_train_2), y_train_2))
#     err_test_2.append(loss(knn_2.predict(X_test_2), y_test_2))

# plt.plot(K_list, err_train_2, '-', label='split #2 train')
# plt.plot(K_list, err_test_2, '-', label='split #2 unseen')
# plt.legend()
# plt.xlabel('K (number of neighbours)')
# plt.ylabel('mean squared error')
# plt.show()


# Plot the mean square error for different K values stored in K_list
K_list = range(1,51)
err_train_1, err_test_1 = [], []
for i, K in enumerate(K_list):
    knn_1 = KNeighborsRegressor(n_neighbors=K, metric = 'minkowski', p = 2)
    knn_1 = knn_1.fit(X_train_1, y_train_1)
    err_train_1.append(loss(knn_1.predict(X_train_1), y_train_1))
    err_test_1.append(loss(knn_1.predict(X_test_1), y_test_1))

plt.plot(K_list, err_train_1, '-', label='split #1 train')
plt.plot(K_list, err_test_1, '-', label='split #1 unseen')

plt.legend()
plt.xlabel('K (number of neighbours)')
plt.ylabel('mean squared error')
plt.show()


















