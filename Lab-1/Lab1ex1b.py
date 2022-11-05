#!/usr/bin/env python3

"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 25.04.2019
"""

# Run with this command
# mpiexec -n 4 python Lab1ex1b.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

### Part (b) Find an average of numbers in a vector ##

#n=10**2
#n=10**5
n=10**7

# Divide vector size(n) by number of workers P(size)
def slices(n, size):
    w = int(n / size)
    return [i*w for i in range(size)][1:]

#BDA lecture slide 2 pg 8
if rank != 0:
    print('Slices are processing on Node: {} '.format(rank))
    avg = comm.recv(source=MPI.ANY_SOURCE, status=status)
    print('worker is doing the tasks..')
    #avg = np.mean(data, dtype=np.float64)
    #avg = np.mean(data,axis=1)
    data = np.mean(avg)
    comm.send(data, dest=0)

else:
    print("Starting root Node 0")
    wt = MPI.Wtime()  # Starting the timer
    #print("start time is..{}".format(wt))

    avg_out = []  # initialize the list
    input_vec = np.random.randint(1, 10, (1, n))

    print("Vector 1 input: {}".format(input_vec))
    avg_vec_slice = np.split(input_vec, slices(len(input_vec), size))
    for i in range(1, size):
        tmp_avg = (avg_vec_slice[i])
        comm.send(tmp_avg, dest=i)

    for i in range(1, size):
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)
        sender = status.Get_source()
        print("Receiving from Node: {}".format(sender))
        avg_out.append(data)

    print("Root node has received all the slices...")
    wt = MPI.Wtime() - wt
    print("Vector average output: {}".format(avg_out))
    print("Total time taken: {}".format(wt))
