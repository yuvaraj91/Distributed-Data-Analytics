#!/usr/bin/env python3

"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 25.04.2019
"""

# Run with this command
# mpiexec -n 4 python Lab1ex3.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

### Exercise 2: Parallel Matrix Vector multiplication using MPI ##

#n=10**2
n=10**3
#n=10**4

# def slices(n, size):
#     w = int(n / size)
#     return [i*w for i in range(size)][1:]

if rank != 0:
    print('H-Slices and V-Slices are processing on Node: {} '.format(rank))
    s1, s2 = comm.recv(source=MPI.ANY_SOURCE, status=status)
    data = np.matmul(s1,s2)
    comm.send(data, dest=0)

else:
    print("Starting root Node 0")
    wt = MPI.Wtime()  # Starting the timer

    matA = np.array([[np.random.randint(1,10) for i in range(n)] for j in range(n)])
    matB = np.array([[np.random.randint(1,10) for i in range(n)] for j in range(n)])
    matC = []

    print("Matrix A input is: {}".format(matA))
    print("Matrix B input is: {}".format(matB))

    matA_slice = np.vsplit(matA, size) # Split matrix A into multiple sub-arrays vertically (column-wise)
    matB_slice = np.hsplit(matB, size) # Split matrix B into multiple sub-arrays horizontally (column-wise)

    for i in range(1, size):
        tmp_out = (matA_slice[i], matB_slice[i])
        # print(task)
        comm.send(tmp_out, dest=i)

    for i in range(1, size):
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)
        sender = status.Get_source()
        print("Receiving from Node: {}".format(sender))
        matC.extend(data)

    print("Root node has received all the slices...")
    wt = MPI.Wtime() - wt
    print("Matrix C output: {}".format(matC))
    print("Total time taken: {}".format(wt))




