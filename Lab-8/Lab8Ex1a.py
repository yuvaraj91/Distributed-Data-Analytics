#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 14.06.2019
"""

from pyspark import SparkContext

sc = SparkContext.getOrCreate()

# Define data as a list
a = ["spark", "rdd", "python", "context", "create", "class"]
b = ["operation", "apache", "scala", "lambda", "parallel", "partition"]

# Convert list to RDDs
rddA = sc.parallelize(a)
rddB = sc.parallelize(b)

# Lambda function for key-value pair mapping, identify based on source RDD
distA = rddA.map(lambda word: (word, 'Rdd_A'))
distB = rddB.map(lambda word: (word, 'Rdd_B'))

## Question 1 ##

ro_join = distA.rightOuterJoin(distB).collect()
print("RIGHT OUTER JOIN: \n", ro_join, "\n")

fo_join = distA.fullOuterJoin(distB).collect()
print("FULL OUTER JOIN: \n", fo_join)

## Question 2 ##

rddC = rddA.union(rddB)  # SQL UNION operation

# https://stackoverflow.com/questions/36559071/how-to-count-number-of-occurrences-by-using-pyspark
S_count= rddC.flatMap(lambda i:list(i)).map(lambda i:i.count('s')).reduce(lambda i,j:i+j)
print("MapReduce - Number of times 's' appears: ", S_count)

# https://stackoverflow.com/questions/51001135/spark-count-number-of-specific-letter-in-rdd-using-aggregate-function
S_count2=rddC.aggregate(0, lambda i, x: i + x.count('s'), lambda i, j: i+j)
print("Aggregate - Number of times 's' appears: ", S_count2)

