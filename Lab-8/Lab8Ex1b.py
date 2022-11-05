#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 14.06.2019
"""

from pyspark.sql.functions import mean, to_timestamp, unix_timestamp, when, col, to_date, datediff, lit, round, stddev
from pyspark.sql import SparkSession
import datetime
from matplotlib import pyplot as plt

spark = SparkSession \
    .builder \
    .appName("Python Spark") \
    .getOrCreate()

# DATAFRAMES
df = spark.read.json('C:/PythonProjects/students.json')

print("Original json dataset: ", "\n")
df.show()

# 1. Replace the null value(s) in column points by the mean of all points.
df = df.na.fill(df.agg(mean('points')).collect()[0][0])

# 2. Replace the null value(s) in column dob and column last name by "unknown" and "--" respectively.
df = df.na.fill('unknown', 'dob')
df = df.na.fill('--', 'last_name')

# 3. Convert all the dates into DD-MM-YYYY format.
# https://stackoverflow.com/questions/39088473/pyspark-dataframe-convert-unusual-string-format-to-timestamp
df2 = df.select('*', to_timestamp(df.dob, 'MMMMM dd, yyyy').alias('date'))
df2 = df2.withColumn('date', when(col('date').isNull(), to_timestamp(df.dob, 'dd MMMMM yyyy')).otherwise(col('date')))
df2 = df2.withColumn('dob', unix_timestamp(col("date"), 'yyyy-MM-dd HH:mm:ss').cast("timestamp"))
df2 = df2.withColumn('dob', to_date('dob', 'dd-MM-yyyy'))
df2 = df2.drop('date')

# 4. Insert a new column age and calculate the current age of all students.
# https://stackoverflow.com/questions/44020818/how-to-calculate-date-difference-in-pyspark
df2 = df2.select('*', round(datediff(lit(datetime.date.today()), col('dob')) / 365, 0).alias('age'))

# 5. If point > 1 standard deviation of all points, then update current point to 20
grades = df2.agg({'points': 'mean'}).collect()[0][0] + df2.agg({'points': 'stddev'}).collect()[0][0]
df2 = df2.withColumn('points', when(col('points') > grades, 20).otherwise(col('points')))

print("Final json dataset: ", "\n")
df2.show()

# 6. Create a histogram on the new points created in the task 5.
points_histogram = df2.select('points').rdd.map(lambda i: i.points).collect()
plt.hist(points_histogram)
plt.title('Points Histogram')
plt.show()
