#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 14.06.2019
"""

from pyspark.sql.functions import mean, to_timestamp, from_unixtime, when, col, unix_timestamp, count
from pyspark.sql.functions import to_date, datediff, lit, round, stddev, dense_rank, udf
from pyspark.sql.types import TimestampType
from pyspark.sql import SparkSession
import datetime
from matplotlib import pyplot as plt
from pyspark.sql.window import Window

spark = SparkSession \
    .builder \
    .appName("Python Spark") \
    .getOrCreate()

df = spark.read.csv('C:/PythonProjects/tags.dat', sep=':')
print("Original tags dataset: ", "\n")
df.show(10)

# Data pre-processing
df = df.drop('_c1', '_c3', '_c5')  # Remove extra NULL columns because we cannot use delimeter '::'
df = df.selectExpr('_c0 as UserID', '_c2 as MovieID', '_c4 as Tag', '_c6 as Timestamp')  # Rename columns
df = df.distinct()  # Drop duplicates
df = df.withColumn('Timestamp', from_unixtime(df.Timestamp).cast(TimestampType()))  # Convert string into timestamp

print("Modified tags dataframe: ", "\n")
df.show(10)

# 1. Separate out tagging sessions for each user.
# https://stackoverflow.com/questions/41661068/group-by-rank-and-aggregate-spark-data-frame-using-pyspark
# https://stackoverflow.com/questions/36725353/applying-a-window-function-to-calculate-differences-in-pyspark
w = Window.partitionBy(df['UserID']).orderBy(df['TimeStamp'].asc(), df['MovieID'].asc(), df['tag'].asc())
# https://stackoverflow.com/questions/44968912/difference-in-dense-rank-and-row-number-in-spark
df = df.withColumn("Rank", dense_rank().over(w))

# Cross join to append new column from timestamp as next tagging time
df.createOrReplaceTempView(
    "tags")  # https://stackoverflow.com/questions/44011846/how-does-createorreplacetempview-work-in-spark/44013035
df = spark.sql("SELECT a.*, b.Timestamp as Next_tag        \
                   FROM tags a                             \
                   left join tags b                        \
                    on a.UserID = b.UserID                 \
                    and a.Rank = b.Rank - 1")

# Calculate time difference between tags
df = df.withColumn('Time_diff', unix_timestamp('Next_tag') - unix_timestamp('Timestamp'))
# Identifier flag if tag is part of a session. Inactive duration of 30 mins (1800 seconds)
df = df.withColumn("Identifier", when(col('Time_diff') < 1800, 1).otherwise(0))

print("Tags dataframe with session identifier: ", "\n")
df.show(10)

df.createOrReplaceTempView("tags")
a = df.alias('a')
b = df.alias('b')

df = spark.sql("SELECT a.*, b.Identifier as prev_identifier        \
                   FROM tags a                         \
                   left join tags b                    \
                          on a.UserID = b.UserID       \
                         and a.Rank = b.Rank + 1       \
                   order by a.UserId, a.Rank")

cnt = 1

# User defined function pyspark
def udf_tag(identifier, prev_identifier):
    global cnt
    if prev_identifier is None:
        cnt = 1
        return cnt  # Initial record
    elif prev_identifier == 1:
        return cnt  # Return same session ID if same session
    elif prev_identifier == 0:
        cnt = cnt + 1
        return cnt  # Return increment session ID if new session
    else:
        return 0  # Unknown session


sess_udf = udf(udf_tag)  # Initialize UDF
df = df.sort("UserID", "Rank")  # Sort (asc) on UserID, followed by Rank
df = df.withColumn('SessionID', sess_udf("identifier", "prev_identifier"))  # Append new column SessionID from UDF
df = df.drop('Next_tag', 'Identifier', 'prev_identifier', 'Time_diff')  # Drop the temp working columns

print("Modified Tags dataframe with each user session: ", "\n")
df.show(10)

# 2. Calculate the frequency of tagging for each user session.
df2 = df.groupby(['UserID', 'SessionID']).count() \
    .select('UserID', 'SessionID', col('count').alias('Freq_tags')) \
    .sort("UserID", "SessionID").cache()

print("Frequency tagging dataframe of UserID: ", "\n")
df2.show(10)

# 3. Find a mean and standard deviation of the tagging frequency of each user.
df3 = df2.groupby('UserID') \
    .agg(mean('Freq_tags').alias('Mean_tagging_freq')
         , count(lit(1)).alias('Sessions_count')
         , stddev('Freq_tags').alias('Std_Dev')) \
    .sort('UserID')

print("Mean and Standard deviation dataframe of UserID: ", "\n")
df3.show(10)

# 4. Find a mean and standard deviation of the tagging frequency for across users.
df4 = df2.agg(mean('Freq_tags').alias('Mean_tagging_freq')
              , count(lit(1)).alias('Sessions_count')
              , stddev('Freq_tags').alias('Std_Dev'))

print("Total Mean and Standard deviation dataframe of UserID: ", "\n")
df4.show(10)

# 5. List of users with a mean tagging frequency within the two standard deviation from the mean frequency of all users.
mean = df4.collect()[0]['Mean_tagging_freq']
std_dev = df4.collect()[0]['Std_Dev']

df5 = df3.filter((col('Mean_tagging_freq') >= (mean - (2 * std_dev))) &
                 (col('Mean_tagging_freq') <= (mean + (2 * std_dev)))) \
    .select('UserID')

print("List of users within 2 std dev from mean: ", "\n")
df5.show(10)
