# -*- coding: utf-8 -*-
"""Pallapu_srinivas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_eir_1PHkuIoktZYh2d5Ey--ouxFiai0

# importing libraries
"""

import pandas as pds
import numpy as np
import matplotlib.pyplot as pyp
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sea

"""# imorting the train dataset


"""

# loading the data
df = pds.read_csv('ALS_TrainingData_2223.csv')
df.head()

"""# check for sum null values"""

df.isnull().sum()

"""# Summery of dataset"""

df.info()

"""# Columns of the train dataset"""

for names in df.columns:
  print(names)

distributiontraindata = df.hist(figsize=(40,30),bins=25, grid=False, zorder=2, rwidth = 0.9, color="green")

"""# Correlation Matrix"""

fig, ax = pyp.subplots(figsize=(24, 18))

corr = df.iloc[:, 1:].corr()
heatmap = sea.heatmap(df.corr(),cbar=True, vmin=-0.5, vmax=0.5, fmt='.2f', annot_kws={'size': 3}, annot=True, square=True)
ax.set_title('Correlation Matrix')
pyp.tight_layout()

print('Column with max correlation:',max(corr))
print('Column with min correlation:',min(corr))

"""# choosing five variables to be clustered"""

kmeans_data = df[['Age_mean', 'ALSFRS_slope','Chloride_median' ,'Potassium_median', 'Sodium_median']]
kmeans_data

"""# PCA transformation of k-means data"""

pca_data = PCA(n_components=2).fit_transform(kmeans_data)
pca_data

"""# K-Means model for 3 clusters"""

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(pca_data)
print(y_predicted)

"""# scatter diagram with three custers and their corresponding centers"""

pyp.scatter(pca_data[y_predicted==0,0], pca_data[y_predicted==0,1], c="yellow", label="cluster 1")    
pyp.scatter(pca_data[y_predicted==1,0], pca_data[y_predicted==1,1], c="red", label="cluster 2")    
pyp.scatter(pca_data[y_predicted==2,0], pca_data[y_predicted==2,1], c="green", label="cluster 3")
pyp.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=120, c="black",marker='*', label="cluster center")
pyp.legend()

"""# model's silhouette score with three clusters"""

labels = km.labels_
metrics.silhouette_score(kmeans_data, labels)

"""# K-Means model for 4 clusters"""

km1 = KMeans(n_clusters=4)
y1_predicted = km1.fit_predict(pca_data)
print(y1_predicted)

"""# scatter diagram with four custers and their corresponding centers"""

pyp.scatter(pca_data[y1_predicted==0,0], pca_data[y1_predicted==0,1], c="red", label="cluster 1")    
pyp.scatter(pca_data[y1_predicted==1,0], pca_data[y1_predicted==1,1], c="blue", label="cluster 2")    
pyp.scatter(pca_data[y1_predicted==2,0], pca_data[y1_predicted==2,1], c="green", label="cluster 3")
pyp.scatter(pca_data[y1_predicted==3,0], pca_data[y1_predicted==3,1], c="pink", label="cluster 4")
pyp.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], s=120, c="black", marker='*',label="cluster center")
pyp.legend()

"""# model's silhouette score with four clusters"""

labels1 = km1.labels_
metrics.silhouette_score(kmeans_data, labels1)

"""# K-Means model for 5 clusters"""

km2 = KMeans(n_clusters=5)
y2_predicted = km2.fit_predict(pca_data)
print(y2_predicted)

"""# scatter diagram with five custers and their corresponding centers"""

pyp.scatter(pca_data[y2_predicted==0,0], pca_data[y2_predicted==0,1], c="red", label="cluster 1")    
pyp.scatter(pca_data[y2_predicted==1,0], pca_data[y2_predicted==1,1], c="blue", label="cluster 2")    
pyp.scatter(pca_data[y2_predicted==2,0], pca_data[y2_predicted==2,1], c="green", label="cluster 3")
pyp.scatter(pca_data[y2_predicted==3,0], pca_data[y2_predicted==3,1], c="pink", label="cluster 4")
pyp.scatter(pca_data[y2_predicted==4,0], pca_data[y2_predicted==4,1], c="indianred", label="cluster 5")

pyp.scatter(km2.cluster_centers_[:,0], km2.cluster_centers_[:,1], s=120, c="black",marker='*',label="cluster center")

"""# model's silhouette score with five clusters"""

labels2 = km2.labels_
metrics.silhouette_score(kmeans_data,labels2)

"""# K-Means model for seven clusters"""

km3 = KMeans(n_clusters=7)
y3_predicted = km3.fit_predict(pca_data)
print(y3_predicted)



"""# scatter diagram with seven custers and their corresponding centers"""

pyp.scatter(pca_data[y3_predicted==0,0], pca_data[y3_predicted==0,1], c="red", label="cluster 1")    
pyp.scatter(pca_data[y3_predicted==1,0], pca_data[y3_predicted==1,1], c="blue", label="cluster 2")    
pyp.scatter(pca_data[y3_predicted==2,0], pca_data[y3_predicted==2,1], c="green", label="cluster 3")
pyp.scatter(pca_data[y3_predicted==3,0], pca_data[y3_predicted==3,1], c="pink", label="cluster 4")
pyp.scatter(pca_data[y3_predicted==4,0], pca_data[y3_predicted==4,1], c="indianred", label="cluster 5")
pyp.scatter(pca_data[y3_predicted==5,0], pca_data[y3_predicted==5,1], c="indianred", label="cluster 6")
pyp.scatter(pca_data[y3_predicted==6,0], pca_data[y3_predicted==6,1], c="yellow", label="cluster 7")
pyp.scatter(km3.cluster_centers_[:,0], km3.cluster_centers_[:,1], s=120, c="black", marker='*',label="cluster center")

"""# model's silhouette score with seven clusters"""

labels3 = km3.labels_
metrics.silhouette_score(kmeans_data, labels3)

"""
# Using the elbow method to find the optimal number of clusters"""

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

pyp.figure(figsize=(8,8))
sea.lineplot(x=range(1, 15), y=wcss, marker='P', color='blue')
pyp.title('Elbow Method')
pyp.xlabel('Number of clusters')
pyp.ylabel('WCSS')
pyp.show()

"""#importing test data"""

df2 = pds.read_csv('ALS_TestingData_78.csv')
df2.head()

"""# check for sum null values"""

df2.isnull().sum()

"""# summary for getting of dataset"""

df2.info()

"""#columns of test data"""

for names in df2.columns:
  print(names)

print('Column with max correlation:',max(corr))
print('Column with min correlation:',min(corr))

"""# choosing five variables to be clustered """

kmeans_data2 = df2[['Age_mean', 'ALSFRS_slope','Chloride_median' ,'Potassium_median', 'Sodium_median']]
kmeans_data2

"""# PCA transformation of k-means data"""

pca_data2 = PCA(n_components=2).fit_transform(kmeans_data2)
pca_data2

"""# K-Means model for 3 clusters"""

km_2 = KMeans(n_clusters=3)
y_predicted2 = km_2.fit_predict(pca_data2)
print(y_predicted2)

"""# scatter diagram with three clusters and their corresponding centers"""

pyp.scatter(pca_data2[y_predicted2==0,0], pca_data2[y_predicted2==0,1], c="yellow", label="cluster 1")    
pyp.scatter(pca_data2[y_predicted2==1,0], pca_data2[y_predicted2==1,1], c="red", label="cluster 2")    
pyp.scatter(pca_data2[y_predicted2==2,0], pca_data2[y_predicted2==2,1], c="green", label="cluster 3")
pyp.scatter(km_2.cluster_centers_[:,0], km_2.cluster_centers_[:,1], s=120, c="black",marker='*', label="cluster center")
pyp.legend()

"""# model's silhouette score with three clusters"""

labels_2 = km_2.labels_
metrics.silhouette_score(kmeans_data2, labels_2)

"""# K-Means model for 5 clusters"""

km_3 = KMeans(n_clusters=5)
y_predicted3 = km_3.fit_predict(pca_data2)
print(y_predicted3)

"""# scatter diagram with five clusters and their corresponding centers"""

pyp.scatter(pca_data2[y_predicted3==0,0], pca_data2[y_predicted3==0,1], c="yellow", label="cluster 1")    
pyp.scatter(pca_data2[y_predicted3==1,0], pca_data2[y_predicted3==1,1], c="red", label="cluster 2")    
pyp.scatter(pca_data2[y_predicted3==2,0], pca_data2[y_predicted3==2,1], c="green", label="cluster 3")
pyp.scatter(pca_data2[y_predicted3==3,0], pca_data2[y_predicted3==3,1], c="blue", label="cluster 4")
pyp.scatter(pca_data2[y_predicted3==4,0], pca_data2[y_predicted3==4,1], c="indianred", label="cluster 5")
pyp.scatter(km_3.cluster_centers_[:,0], km_3.cluster_centers_[:,1], s=120, c="black",marker='*', label="cluster center")
pyp.legend()

"""# model's silhouette score with five clusters"""

labels_3 = km_3.labels_
metrics.silhouette_score(kmeans_data2, labels_3)

"""# K-Means model for 5 clusters"""

km_4 = KMeans(n_clusters=7)
y_predicted4 = km_4.fit_predict(pca_data2)
print(y_predicted4)

"""# scatter diagram with seven clusters and their corresponding centers"""

pyp.scatter(pca_data2[y_predicted4==0,0], pca_data2[y_predicted4==0,1], c="yellow", label="cluster 1")    
pyp.scatter(pca_data2[y_predicted4==1,0], pca_data2[y_predicted4==1,1], c="red", label="cluster 2")    
pyp.scatter(pca_data2[y_predicted4==2,0], pca_data2[y_predicted4==2,1], c="green", label="cluster 3")
pyp.scatter(pca_data2[y_predicted4==3,0], pca_data2[y_predicted4==3,1], c="blue", label="cluster 4")
pyp.scatter(pca_data2[y_predicted4==4,0], pca_data2[y_predicted4==4,1], c="indianred", label="cluster 5")
pyp.scatter(pca_data2[y_predicted4==5,0], pca_data2[y_predicted4==5,1], c="skyblue", label="cluster 6")
pyp.scatter(pca_data2[y_predicted4==6,0], pca_data2[y_predicted4==6,1], c="orange", label="cluster 7")
pyp.scatter(km_3.cluster_centers_[:,0], km_3.cluster_centers_[:,1], s=120, c="black",marker='*', label="cluster center")
pyp.legend()

"""
Using the elbow method to find the optimal number of clusters"""

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 15):
    kmeans3 = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans3.fit(df2)
    wcss.append(kmeans3.inertia_)

pyp.figure(figsize=(8,8))
sea.lineplot(x=range(1, 15), y=wcss, marker='P', color='blue')
pyp.title('Elbow Method')
pyp.xlabel('Number of clusters')
pyp.ylabel('WCSS')
pyp.show()

"""
**Conclusion**: In this K-means clustering model i can conclude that cluster 3 is having a high model in it. You can see in a above value in the given plot graph. There is highest value is given and in the elbow method also showing the number of clusters here."""