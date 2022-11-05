#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 29.04.2019
"""
# Run with this command
# mpiexec -n 4 python Lab2Ex2.py

from mpi4py import MPI
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print('Number of workers: {}'.format(size))
    img = cv.imread('img.png', 0)  # Import image as greyscale only
    #cv.namedWindow('KLCC', cv.WINDOW_NORMAL)
    #cv.imshow('KLCC',img)
    data=np.array(img,dtype='i')
    data=np.array_split(data,size)
else:
    data=None

wt = MPI.Wtime()
# Scatter split arrays to workers
data=comm.scatter(data,root=0)
#print("Data {} sent to node {}".format(data,rank))

# Calculate greyscale intensity pixels
pixels=np.zeros(256,dtype='i')
for i in range(0, len(data)):
    for j in range (0,len(data[i])):
        pixels[data[i][j]] += 1

# Return sum of data to root node
gs = comm.reduce(pixels, op=MPI.SUM, root=0)
print('Total time taken: {}' .format(MPI.Wtime() - wt))
if rank == 0:
    print("Greyscale frequencies: {}".format(gs))
    bins = np.arange(256, dtype='i')
    plt.bar(bins, gs)
    plt.title('Frequency histogram of greyscale using MPI')
    plt.ylabel('Intensity')
    plt.xlabel('Pixel')
    plt.show()

# Maintain image windows
# while True:
#     k = cv.waitKey(0) & 0xFF
#     if k == 27: break             # ESC key to exit
# cv.destroyAllWindows()
