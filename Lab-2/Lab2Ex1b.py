#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 01.05.2019
"""
# Run with this command
# mpiexec -n 4 python Lab2Ex1b.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

n = 10**3
#n=10**5
#n=10**7

# Recursive doubling algorithm
def EsendAll(data):
    print('Input Array: {}...'.format(data[:5]))
    destA = 2 * int(rank) + 1
    destB = destA+1
    if destA < size:
        comm.send(data, dest=destA)
        print("Sending array from node {} to node {}".format(rank, destA))
    if destB < size:
        comm.send(data, dest=destB)
        print("Sending array from node {} to node {}".format(rank, destB))
    comm.Barrier()


wt = MPI.Wtime()
if rank == 0:
    print('Number of workers: {}'.format(size))
    data = np.arange(n, dtype='i')
    EsendAll(data)
    print("Root node {} sending array".format(rank))
else:
    recvProc = int((rank - 1) / 2)
    print("recvProc is {}".format(recvProc))
    data = comm.recv(source=recvProc)
    EsendAll(data)

comm.Barrier()
print("Total Time taken: {}".format(MPI.Wtime() - wt))
