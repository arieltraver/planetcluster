#SOUCE: https://machinelearningmastery.com/clustering-algorithms-with-python/
#SOURCE: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataset

#GAUSSIAN MIXTURE
#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
from numpy import unique
from numpy import where
from sklearn.cluster import Birch, DBSCAN, MiniBatchKMeans, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
from sklearn.neighbors import kneighbors_graph
import os
import pandas as pd
import numpy as np




def create_frame(filename):
    frame = pd.read_csv(filename)
    return frame

#1
df0 = create_frame("paper.csv")
knn_graph = kneighbors_graph(df0, 6)
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model0 = AgglomerativeClustering(n_clusters=2)
# fit the model
yhat = model0.fit_predict(df0)
# assign a cluster to each example
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    df_filtered = df0.drop(row_ix[0])
    # create scatter of these samples
    pyplot.scatter(df_filtered["st_met"], df_filtered["pl_f"])
    pyplot.ylabel("planet mantle and core fe (fraction)")
    pyplot.xlabel("stellar metallicity")
# show the plot
pyplot.show()

#2
# Gaussian Mixture
df = create_frame("paper.csv")
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model = GaussianMixture(n_components=2)
# fit the model
model.fit(df)
# assign a cluster to each example
yhat = model.predict(df)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    df_filtered = df.drop(row_ix[0])
    # create scatter of these samples
    pyplot.ylabel("planet mantle and core fe (fraction)")
    pyplot.title("gaussian mixture")
    pyplot.xlabel("stellar metallicity")
    pyplot.scatter(df_filtered["st_met"], df_filtered["pl_f"])
# show the plot
pyplot.show()

#3
# Birch
df2 = create_frame("paper.csv")
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model2 = Birch(threshold=0.01, n_clusters=2)
# fit the model
model2.fit(df2)
# assign a cluster to each example
yhat = model2.predict(df2)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    df_filtered = df2.drop(row_ix[0])
    # create scatter of these samples
    pyplot.scatter(df_filtered["st_met"], df_filtered["pl_f"])
    pyplot.title("Birch")
    pyplot.ylabel("planet mantle and core fe (fraction)")
    pyplot.xlabel("stellar metallicity")
# show the plot
pyplot.show()

#4
# DBScan
df3 = create_frame("paper.csv")
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model3 = DBSCAN(eps=0.2, min_samples=4)
# assign a cluster to each example
yhat = model3.fit_predict(df3)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat != cluster)
    df_filtered = df3.drop(row_ix[0])
    # create scatter of these samples
    pyplot.scatter(df_filtered["st_met"], df_filtered["pl_f"])
    pyplot.ylabel("planet mantle and core fe (fraction)")
    pyplot.xlabel("stellar metallicity")
# show the plot
pyplot.show()

# K Means
df4 = create_frame("paper.csv")
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model4 = MiniBatchKMeans(n_clusters=2, n_init=50)
model4.fit(df4)
# assign a cluster to each example
yhat = model4.predict(df4)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat != cluster)
    df_filtered = df4.drop(row_ix[0])
    # create scatter of these samples
    pyplot.scatter(df_filtered["st_met"], df_filtered["pl_f"])
    pyplot.ylabel("planet mantle and core fe (fraction)")
    pyplot.xlabel("stellar metallicity")
# show the plot
pyplot.show()

# MeanShift
df5 = create_frame("paper.csv")
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model3 = MeanShift()
# assign a cluster to each example
yhat = model3.fit_predict(df5)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat != cluster)
    df_filtered = df5.drop(row_ix[0])
    # create scatter of these samples
    pyplot.scatter(df_filtered["st_met"], df_filtered["pl_f"])
    pyplot.ylabel("planet mantle and core fe (fraction)")
    pyplot.xlabel("stellar metallicity")
# show the plot
pyplot.show()