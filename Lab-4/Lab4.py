#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 15.05.2019
"""
# Run with this command
# mpiexec -n 4 python Lab4.py

from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import glob
import random
from math import sqrt
from copy import deepcopy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
epochs = 1  # Initialize counter of while loop


# Import dataset
def ImportData(): # Concatenate all input files into one file for easier processing
    read_files = glob.glob("C:/PythonProjects/dataset/*.txt")
    with open("dataset.txt", "wb") as outfile:  # https://realpython.com/working-with-files-in-python/
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

    x_df = []
    y_df = []
    x, y = load_svmlight_file(
        "dataset.txt")  # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
    x_df.append(pd.DataFrame(x.todense()))
    y_df.append(y)
    x_df = pd.concat(x_df).fillna(0)  # https://stackoverflow.com/questions/48956789/converting-nan-in-dataframe-to-zero
    # Must normalize X and Y before read to DataFrame
    x_df = pd.DataFrame(normalize(x_df))
    y_df = pd.DataFrame(np.concatenate(np.array(normalize(y_df))))
    return x_df, y_df


# Function to calculate yhat, the predicted best fit
def predict_yhat(data, beta):
    y_hat = np.zeros(data.shape[0])  # Initial numpy array to store the result
    for i in range(0, data.shape[1]):  # Range of features (columns)
        y_hat = np.dot(data[i],(beta[i]))
    return pd.DataFrame(y_hat)


# SGD function to run in parallel to calculate coefficient beta of training dataset
# https://medium.com/analytics-vidhya/linear-regression-in-python-from-scratch-24db98184276
def SGD_func(X, Y, pred_Y, beta, learning_rate, sample):
    for i in random.sample(range(0, X.shape[0]), sample):  # Calculate for random row (training data)
        err = (pred_Y.iloc[i] - Y.iloc[i])  # Gets rows at particular positions in the index
        for j in range(0, X.shape[1]):  # Range of features (columns)
            beta[j] -= (learning_rate * err * X.iloc[i][j])  # Weights coefficient = learning rate*error*training datarows
    return beta


def cost_func(yTest_data, y_hat_data):
    # https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python
    rmse = sqrt(((yTest_data - y_hat_data) ** 2).mean())
    return rmse


# Function to split training data for workers
def slices(xTrain, yTrain):
    xTrain_slice = np.array_split(xTrain, size)
    yTrain_slice = np.array_split(yTrain, size)
    return xTrain_slice, yTrain_slice


if rank == 0:
    print('Number of workers: ', size)
    # Read data
    print("Importing dataset...")
    x_df, y_df = ImportData()
    # Split the targets into training/testing sets - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    xTrain, xTest, yTrain, yTest = train_test_split(x_df, y_df, test_size=0.3)  # 70% training data, 30% test data
    #xTrain, xTest, yTrain, yTest = train_test_split(x_df, y_df, test_size=0.3,random_state=1)  # 70% training data, 30% test data
    # https://stackoverflow.com/questions/42593104/convert-list-into-a-pandas-data-frame
    # Easier processing
    xTrain = pd.DataFrame(np.array(xTrain))
    xTest = pd.DataFrame(np.array(xTest))
    yTrain = pd.DataFrame(np.array(yTrain))
    yTest = pd.DataFrame(np.array(yTest))

    # Call slices function to slice training dataset
    xTrain_slice, yTrain_slice = slices(xTrain, yTrain)

    # Call y_hat prediction function for intial predicted y
    y_hat = predict_yhat(xTest, np.zeros(xTrain.shape[1]))  # Number of features (columns) of training dataset
    # print("Initial predicted Y_hat: ", y_hat)

    # Call Root Mean Squared Error function for initial error
    rmse_val = cost_func(yTest, y_hat)
    print("Initial RMSE: ", rmse_val)
else:
    xTrain_slice = None
    yTrain_slice = None
    xTest = None
    yTest = None
    rmse_val = None

# Scatter training slices to all workers
xTrain_slice = comm.scatter(xTrain_slice, root=0)
yTrain_slice = comm.scatter(yTrain_slice, root=0)

# Set beta coefficients to zero for initial run
beta = np.zeros(xTrain_slice.shape[1])

wt = MPI.Wtime()
# for i in range(0, epochs):
#while true
while epochs < 13:
    # Workers calculate new predicted y_hat
    error = predict_yhat(xTrain_slice, beta)
    # Workers calculate new beta value
    beta = SGD_func(xTrain_slice, yTrain_slice, error, beta, learning_rate=0.01, sample=1000)
    # Gather results of beta coefficients, always must be in Pandas Dataframe type
    gather = pd.DataFrame(comm.gather(beta, root=0))
    comm.barrier()  # Prevent deadlock issue with multiple workers trying to send back data
    # Validate model on test data
    if rank == 0:
        prev_rmse_val = deepcopy(rmse_val)  # Store current RMSE
        beta = gather.mean()  # Average beta coefficients
        error = predict_yhat(xTest, beta)  # Run on test data for prediction
        rmse_val = cost_func(yTest, error) 
        print("Previous RMSE", prev_rmse_val)
        print("Epoch #: {}, Current RMSE: {}".format(epochs, rmse_val))
        diff = rmse_val - prev_rmse_val
    else:
        rmse_val = None
        prev_rmse_val = None
        diff = None
    print('Current time for Process #{}: {} secs.'.format(rank, MPI.Wtime() - wt))

    diff = comm.bcast(diff,root=0) # Broadcast difference result to all workers
    if diff >= 0:
        print("RMSE limit reached, exiting loop...")
        print("Last epoch #:", epochs)
        break  # Exit while loop

    epochs += 1

for i in range(1, size):
    if i == size - 1:
        print("Total Time taken: {}".format(MPI.Wtime() - wt))
