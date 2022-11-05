#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 29.04.2019
"""
# Run with this command
# mpiexec -n 4 python Lab2Ex1a.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

#n = 10**3
#n=10**5
n=10**7

def NsendAll(data):
    print('Number of workers: {}'.format(size))
    print('Input Array: {}...'.format(data[:5]))
    for i in range(1, size):
        comm.send(data, dest=i)
        print("Array sent to node {}".format(i))
        if i == size - 1:
            print("Total Time taken: {}".format(MPI.Wtime() - wt))


if rank == 0:
    wt = MPI.Wtime()
    data = np.arange(n, dtype='i')
    NsendAll(data)
else:
    data = comm.recv(source=0)
comm.Barrier()
