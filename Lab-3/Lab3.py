#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 08.05.2019
"""
# Run with this command
# mpiexec -n 4 python Lab3.py

from mpi4py import MPI
import numpy as np
import pandas as pd
from copy import deepcopy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

k = 3  # Number of clusters set manually
iters = 0
np.random.seed(1)  # Set seed for debugging and validation purpose, seed != 0

def initCenters(k, n):
    # centers = [np.random.randint(np.amin(data), np.amax(data)) for _ in range(k)]
    centers = np.zeros((k, n), dtype="float64")  # Init centroids array
    return centers


# Calculate the euclidean distance between 2 vectors
def computeDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # sqrt((x1 - x2)^2 - (y1 - y2)^2)


# Update the centroids of clusters based on new assignments
# Returns the data rows belonging to each cluster, and the count of each data row for each cluster
def computeCenter(data, clusters):
    nCluster = np.zeros(k, dtype="float64")  # Init vector for clusters count
    sumCluster = np.zeros((k, n), dtype="float64")  # Init matrix for sum of each row in a cluster
    for i in range(data.shape[0]):  # https://github.com/andrewxiechina/DataScience/blob/master/K-Means/K-MEANS.ipynb
        c = clusters[i]  # Cluster of each data row
        sumCluster[c] += data[i] # Sum of cluster rows
        nCluster[c] += 1  # Cluster count
    return sumCluster, nCluster  # return cluster sum and cluster count


if rank == 0:
    df = pd.read_csv("Absenteeism_at_work.csv", delimiter=";", usecols=range(2, 21))
    df = df.values[:, 0:19].astype("float64")

    m = np.shape(df)[0]  # Number of rows / training data
    n = np.shape(df)[1]  # Number of cols / features

    data=np.array_split(df,size)
    centroids = initCenters(k, n)

    # https://github.com/andrewxiechina/DataScience/blob/master/K-Means/K-MEANS.ipynb
    index_arr = np.random.choice(m, k, replace=False)  # Array of k randoms numbers for row count

    for i in range(k):
        centroids[i] = df[index_arr[i]] # Generate centroids array based on k

    meansRank = np.zeros([size, n * k], dtype="float64")  # Init array for buffer cluster means
    countsRank = np.zeros([size, k], dtype="float64")  # Init array for buffer cluster counts
else:
    data = None
    #centroids=None
    centroids = np.zeros((k, 19), dtype="float64")
    meansRank = None
    countsRank = None
m = 740  # Number of rows / training data
n = 19  # Number of cols / features

data = comm.scatter(data, root=0)  # Scatter data array to worker nodes
comm.Bcast(centroids, root=0)  # Broadcast centroid array to all worker nodes

wt = MPI.Wtime()

while True:
    rows = data.shape[0]  # Len of data rows
    dist_mat = np.zeros((rows, k))  # Init distance matrix

    for i in range(rows):
        for j in range(k):
            # Compute distance between data rows and centroids
            dist_mat[i][j] = computeDistance(data[i], centroids[j])

    cluster_vec = np.argmin(dist_mat, axis=1)  # Reshape matrix to calculate along rows

    meansCluster, nCluster = computeCenter(data, cluster_vec)
    meansCluster = meansCluster.flatten()  # Reshape sum to 1-D array

    comm.Gather(meansCluster, meansRank, root=0)  # Gather cluster sums from worker nodes
    comm.Gather(nCluster, countsRank, root=0)  # Gather cluster counts from worker nodes

    if rank == 0:
        # http://benalexkeen.com/k-means-clustering-in-python/
        # Sum of counts per cluster received from each worker
        prev_centroid = deepcopy(centroids)  # Store current centroid
        total_clusters = np.zeros((k, 1), dtype="float64")  # Init array for data rows from workers

        for i in range(size):
            meansCluster = meansRank[i].reshape(k,n)  # Reshape to k*n matrix
            nCluster = countsRank[i].reshape(k, 1) # Reshape to 1-D array
            centroids += meansCluster  # Total sums from each worker
            total_clusters += nCluster  # Total counts from each cluster

        centroids = np.divide(centroids, total_clusters)  # Calculate centroid (mean) to send back
        #centroids = np.mean(centroids, axis=0)
        print("Current clusters arrangement ", total_clusters,"for iteration",iters)
        # Check if any changes between old and new centroids assignment
        diff = np.sum([centroids[i] - prev_centroid[i] for i in range(len(centroids))])
        print('Computed centroid changes: {}'.format(diff))
    else:
        diff=None

    diff=comm.bcast(diff,root=0) # Broadcast difference result to all workers
    if diff == 0:
        break  # Exit while loop

    comm.Bcast(centroids, root=0)  # Broadcast centroids array back to all workers for each iteration
    iters += 1

if rank == 0:
    print("Number of K-clusters",k)
    print("Final centroids after {} iterations: {}".format(iters,centroids))
    print('Total time taken: {}' .format(MPI.Wtime() - wt))
