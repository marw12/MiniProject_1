# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', 500)

# Importing the dataset
dataset_1 = pd.read_csv('2020_US_weekly_symptoms_dataset.csv')  #Search Trends dataset

#Cleaning the datasets

dataset_1 = dataset_1.dropna(axis='columns', how='all')  #removes columns with all NaN values
dataset_1 = dataset_1.dropna(axis='rows', how='all')  #removes rows with all NaN values

print(list(dataset_1.columns) )

#convert dataframe into numpy array for calculations
X = dataset_1.iloc[:, 4:].values

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


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

# kmeans_kwargs = {
#     "init": "random",
#     "n_init": 10,
#     "max_iter": 300,
#     "random_state": 42,
# }

# wcss = []
# for i in range(1, 81):
#     kmeans = KMeans(n_clusters = i, **kmeans_kwargs)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 81), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Cost')
# plt.show()

# A list holds the silhouette coefficients for each k
# silhouette_coefficients = []

# # Notice you start at 2 clusters for silhouette coefficient
# for k in range(2, 81):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(X)
#     score = silhouette_score(X, kmeans.labels_)
#     silhouette_coefficients.append(score)
    
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 81), silhouette_coefficients)
# plt.xticks(range(2, 81))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()

#Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 16, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#Reduce dimension of X to 2D array for visualizing 
tsne_obj= tsne.fit_transform(X)

cluster_result = pd.DataFrame({'open_covid_region_code':dataset_1.iloc[:, 0].values, 'date':dataset_1.iloc[:, 5].values, 'cluster':y_kmeans})
tsne_df = pd.DataFrame({'X':tsne_obj[:,0], 'Y':tsne_obj[:,1], 'cluster':y_kmeans})

#plot the clusters
sns.scatterplot(x="X", y="Y", data=tsne_df, hue="cluster", palette=['purple','red','orange','brown','blue',
                       'dodgerblue','green','lightgreen','darkcyan', 'black', 'pink', 'tan', 'aqua', 'darkgray', 'yellow', 'lime'],
              legend='full');

    