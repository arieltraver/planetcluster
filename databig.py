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

#2
# Gaussian Mixture
df = create_frame("paper.csv")

#DROP errors above 0.3
df.drop(df[df['pl_denserr1'] > 0.3].index)
df.drop(df[df['pl_denserr2'] > 0.3].index)
df.drop(df[df['st_meterr'] >0.3].index)

#drop non measured columns
df.drop(["pl_denserr1","pl_denserr2","pl_denslim","st_meterr1","st_meterr2","st_metlim","pl_rade","pl_radeerr1","pl_radeerr2","pl_radelim","pl_bmasse","pl_bmasseer1","pl_bmasseer2","pl_controv_flag"],axis=1)

#DROP NULL values
df = df.dropna() 


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
    pyplot.ylabel("planet density")
    pyplot.title("gaussian mixture")
    pyplot.xlabel("stellar metallicity")
    pyplot.scatter(df_filtered["st_met"], df_filtered["pl_f"])
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
    pyplot.ylabel("planet density")
    pyplot.xlabel("stellar metallicity")
# show the plot
pyplot.show()