#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 31.05.2019
"""
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def generate_data():
    # https://fatihsarigoz.com/tag/python-machine_learning-auto-mpg-dataset.html
    filename = "auto-mpg.data"
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin',
                    'name']
    df = pd.read_csv(filename, delim_whitespace=True, header=None, na_values="?", names=column_names)
    df = df.drop('name', axis=1)
    df = df.dropna()

    # origin = df.pop('origin')
    # df['USA'] = (origin == 1)*1.0
    # df['Europe'] = (origin == 2)*1.0
    # df['Japan'] = (origin == 3)*1.0

    # Ont-hot encoding for category data
    df['origin'] = df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
    df = pd.get_dummies(df, columns=['origin'])
    x_data = df.drop('mpg', axis=1)
    y_data = df[['mpg']]  # Continuous target variable : mpg
    x_data = pd.DataFrame(normalize(x_data))
    y_data = pd.DataFrame(normalize(y_data))

    # Test/Train split of 90%/10%
    xTrain, xTest, yTrain, yTest = train_test_split(x_data, y_data, test_size=0.1, random_state=0)

    # print(xTrain.shape)  # 352x9
    # print(xTest.shape)  # 40x9
    # print(yTrain.shape)  # 352x1
    # print(yTest.shape)  #40x1

    nExamples = xTrain.shape[0]
    nFeatures = xTrain.shape[1]
    return xTrain, xTest, yTrain, yTest, nExamples, nFeatures


def LinearRegression():
    y_pred = tf.add(tf.matmul(X, W), b)
    loss = tf.reduce_sum(tf.abs(y_pred - Y)) / nExamples  # Mean Absolute Error
    #loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, Y))) # Mean Squared Error
    #loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, Y)))) # Root Mean Squared Error
    return loss


xTrain, xTest, yTrain, yTest, nExamples, nFeatures = generate_data()

X = tf.placeholder(tf.float32, [None, nFeatures])
Y = tf.placeholder(tf.float32, [None, 1])

# We are learning W and b over the epochs
W = tf.get_variable(name='Weight', dtype=tf.float32, shape=([nFeatures, 1]), initializer=tf.zeros_initializer())
b = tf.get_variable(name='Bias', dtype=tf.float32, shape=([1]), initializer=tf.zeros_initializer())
loss = LinearRegression()

# Batch gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(loss)

epochs = 50000
display_step = 5000  # Display every 5000s output

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # https://www.edyoda.com/course/1429
    for e in range(epochs):
        _, c = sess.run([optimizer, loss], feed_dict={X: xTrain, Y: yTrain})  # Get loss value
        if (e + 1) % display_step == 0:
            print('Epoch #:', '%d' % (e + 1), 'Loss =', '{:.9f}'.format(c))

    print("Training completed.....")

    # Storing necessary values to be used outside the Session
    # https://www.geeksforgeeks.org/linear-regression-using-tensorflow/
    training_cost = sess.run(loss, feed_dict={X: xTrain, Y: yTrain})
    weight = sess.run(W)
    bias = sess.run(b)
    print("Training cost =", training_cost, '; ' "W =", weight, '; ' "b =", bias)

    # Model prediction on test data
    print("Testing result.....")
    test_loss = LinearRegression()  # Same function as above
    testing_cost = sess.run(test_loss, feed_dict={X: xTest, Y: yTest})
    print("Testing cost:", testing_cost)
