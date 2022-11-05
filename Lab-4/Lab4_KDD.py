#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 15.05.2019
"""
# Run with this command
# mpiexec -n 4 python Lab4_KDD.py

from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import random
from math import sqrt
from copy import deepcopy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
epochs = 1  # Initialize counter of while loop


# Import dataset
def ImportData():
    df = pd.read_csv("cup98LRN.txt", sep=',', header=0, keep_default_na=False,
                     skip_blank_lines=True, low_memory=False, dtype='a', na_filter=True)
    # df.dropna(inplace = True)
    #x_df = []
    #y_df = []
    for col in df.columns:  # https://stackoverflow.com/questions/28910851/python-pandas-changing-some-column-types-to-categories
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    # Drop unwanted columns based on dataset description
    # https://github.com/rebordao/kdd98cup
    X = df.drop(['OSOURCE', 'TARGET_D', 'TARGET_B'], axis=1)
    Y = df.loc[:, 'TARGET_D'] # Target column

    Y = np.expand_dims(Y, axis=0) # https://stackoverflow.com/questions/17394882/add-dimensions-to-a-numpy-array
    X = X.iloc[:, 76:300]  # dropping non-relevant dataset
    # X = np.array(X)

    x_df = pd.DataFrame(normalize(X))
    y_df = pd.DataFrame(np.concatenate(np.array(normalize(Y))))
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
while epochs < 21:
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

