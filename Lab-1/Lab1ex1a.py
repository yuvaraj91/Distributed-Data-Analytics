#!/usr/bin/env python3

"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 25.04.2019
"""

# Run with this command
# mpiexec -n 4 python Lab1ex1a.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

### Part (a) Add two vectors and store results in a third vector ##

#n=10**2
n=10**5
#n=10**7

# Divide vector size(n) by number of workers P(size)
# slices = int(n / size)
def slices(n, size):
    w = int(n / size)
    return [i*w for i in range(size)][1:]

# print("slices is {}".format(slices))

print("Starting program")

#BDA lecture slide 2 pg 8
if rank != 0:
    print('Slices are processing on Node: {} '.format(rank))
    s1, s2 = comm.recv(source=MPI.ANY_SOURCE, status=status)
    data = s1+s2
    comm.send(data, dest=0)  # Adding the slices and send back to Node 0

else:
    print("Starting root Node 0")
    wt = MPI.Wtime()  # Starting the timer
    #print("start time is..{}".format(wt))

    v1 = np.random.randint(1, 10, (1, n))
    v2 = np.random.randint(1, 10, (1, n))
    #v3 = np.empty(n, dtype=int)
    v3 = []

    print("Vector 1: {}".format(v1))
    print("Vector 2: {}".format(v2))

    # Split into equal sub-array by number of slices
    # v1_slice = np.split(v1, len(slices))
    # v2_slice = np.split(v2, len(slices))
    v1_slice = np.split(v1, slices(len(v1), size))
    v2_slice = np.split(v2, slices(len(v2), size))

    # Sending slices to the worker nodes
    for i in range(1, size):
        # v3 = v1_slice[i]+(v2_slice[i])
        tmp_v3 = (v1_slice[i], v2_slice[i])
        print("V3 current loop is ..{}".format(tmp_v3))
        comm.send(tmp_v3, dest=i)
        print("Sending slices...")

    # Receiving slices from worker nodes
    for i in range(1, size):
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)
        sender = status.Get_source()
        print("Receiving from Node: {}".format(sender))
        #np.append(v3, data)
        v3.append(data)

    print("Root node has received all the slices...")
    wt = MPI.Wtime() - wt
    print("Vector 3 output: {}".format(v3))
    print("Total time taken: {}".format(wt))
