#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 14.06.2019
"""

from pyspark.sql.functions import mean as _mean_, to_timestamp, from_unixtime, when, col, unix_timestamp, count
from pyspark.sql.functions import to_date, datediff, lit, round, stddev, dense_rank, udf, max as _max_
from pyspark.sql.functions import min as _min_, count as _count_, explode_outer, split
from pyspark.sql.types import TimestampType
from pyspark.sql import SparkSession
import datetime
from matplotlib import pyplot as plt
from pyspark.sql.window import Window

spark = SparkSession \
    .builder \
    .appName("Python Spark") \
    .getOrCreate()

# Read movie data into python
df_ratings = spark.read.csv('C:/PythonProjects/ratings.dat', sep=':')
df_movie = spark.read.csv('C:/PythonProjects/movies.dat', sep=':')

# Data pre-processing
# http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
df_ratings = df_ratings.drop('_c1', '_c3', '_c5')  # Drop NULL columns
df_ratings = df_ratings.selectExpr('_c0 as UserID', '_c2 as MovieID', '_c4 as Rating', '_c6 as Timestamp')  # Rename columns
df_ratings = df_ratings.withColumn('Timestamp', from_unixtime(df_ratings.Timestamp).cast(TimestampType()))
df_ratings = df_ratings.distinct()  # Drop duplicates

df_movie = df_movie.drop('_c1', '_c3')  # Drop NULL columns
df_movie = df_movie.selectExpr('_c0 as MovieID', '_c2 as Title', '_c4 as Genres')  # Rename columns
df_movie = df_movie.distinct()  # Drop duplicates

print("Ratings dataframe: ", "\n")
df_ratings.show(10)
print("Movies dataframe: ", "\n")
df_movie.show(10)

# https://docs.databricks.com/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html

# 1. Find the movie title which has the maximum average ratings?
df_max = df_ratings.groupby('MovieID')                                   \
                      .agg(_mean_('Rating').alias('Max_avg_rating'))             \
                      .join(df_movie, df_ratings.MovieID == df_movie.MovieID)   \
                      .select(df_movie.MovieID,'Title', 'Max_avg_rating')
print("Movies with maximum avg rating: ", "\n")
df_max.filter(col('Max_avg_rating') == df_max.agg(_max_('Max_avg_rating')).collect()[0][0]).show(10)


# 2. User who has assign the lowest average ratings among all the users the number of ratings greater than 40?
df_low = df_ratings.groupby('UserID').agg(_mean_('Rating').alias('Max_avg_rating'),_count_(lit(1)).alias('Total_ratings'))           \
                      .filter('Total_ratings >= 40')

lowest_rating = df_low.agg(_min_('Max_avg_rating'))
print("Users with with lowest rating assign: ", "\n")
df_low.filter(col('Max_avg_rating') == lowest_rating.collect()[0][0]).show(10)


# 3. Find the movie genre with the highest average ratings?
# http://files.grouplens.org/datasets/movielens/ml-latest-README.html
genre = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy',
         'Film-Noir', 'Horror', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

df_genre = df_ratings.join(df_movie, df_ratings.MovieID == df_movie.MovieID).select(df_movie.Genres, 'Rating')
# https://stackoverflow.com/questions/39739072/spark-sql-how-to-explode-without-losing-null-values
print("Movie genre with highest avg rating: ", "\n")
df_genre.withColumn("Genres", explode_outer(split('Genres', "[|]")))           \
           .filter(col('Genres').isin(genre))                           \
           .groupBy('Genres')                                               \
           .agg(_mean_('Rating').alias('Max_rating'))                        \
           .orderBy(col('Max_rating').desc())                               \
           .show(1)



