#SOUCE: https://machinelearningmastery.com/clustering-algorithms-with-python/
#SOURCE: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataset

#GAUSSIAN MIXTURE
#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
#from sklearn.mixture import DBScan
from matplotlib import pyplot
import os
import pandas as pd
import numpy as np

def create_frame(filename):
    frame = pd.read_csv(filename)
    return frame

# define dataset
df = create_frame("metal_vs_density.csv")
df2 = df.drop(["pl_denserr1","pl_denserr2","pl_denslim","st_meterr1","st_meterr2","st_metlim"],axis=1)
df2 = df2.dropna().reset_index()
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model = GaussianMixture(n_components=2)
# fit the model
model.fit(df2)
# assign a cluster to each example
yhat = model.predict(df2)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    df_filtered = df2.drop(row_ix[0])
    # create scatter of these samples
    pyplot.scatter(df_filtered["pl_dens"], df_filtered["st_met"])
# show the plot
pyplot.show()

'''
#DBSCAN
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# TODO HERE: transform the planet data from a csv into something usable here.
# define the model
model = DBSCAN(eps=0.30, min_samples=9)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
'''