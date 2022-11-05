#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 27.05.2019
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize


def generate_data():
    noise = np.random.normal(0.0, 50, [1000,1]).astype(np.float32)
    x_data = np.random.uniform(0, 1000, [1000,1]).astype(np.float32)
    y_data = 0.5 * x_data + 2 + noise.astype(np.float32)
    #x_data = normalize(x_data, axis=0)
    #y_data = normalize(y_data, axis=0)

    # Test/Train split of 90%/10%
    xTrain, xTest, yTrain, yTest = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
    # Plot ground truth values, distinguish training and testing dataset
    plt.scatter(xTrain, yTrain, s=0.9, label='Train data')
    plt.scatter(xTest, yTest, s=0.9, label='Test data')
    return xTrain, xTest, yTrain, yTest


def LinearRegression():
    #https://stackoverflow.com/questions/56401346/mean-absolute-error-in-tensorflow-without-built-in-functions
    y_pred = tf.add(tf.multiply(X, W), b)
    loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, Y)))
    return loss

# TensorFlow placeholders for feed dictionary
# X=tf.placeholder(shape=(1000,1),dtype=tf.float32)
# Y=tf.placeholder(shape=(1000,1),dtype=tf.float32)
X = tf.placeholder(tf.float32, [None,1])
Y = tf.placeholder(tf.float32, [None,1])

# We are learning W and b over the epochs
W = tf.get_variable(name='Weight', dtype=tf.float32, shape=(), initializer=tf.zeros_initializer())
b = tf.get_variable(name='Bias', dtype=tf.float32, shape=(), initializer=tf.zeros_initializer())

# We are using batch Gradient Descent
loss = LinearRegression()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(loss)
xTrain, xTest, yTrain, yTest=generate_data()

epochs = 100
display_step = 10  # Display every 10s output

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # https://www.edyoda.com/course/1429
    for e in range(epochs):
        _, c = sess.run([optimizer, loss], feed_dict={X: xTrain, Y: yTrain})  # Get loss value
        if (e + 1) % display_step == 0:
            print('Epoch #:', '%d' % (e + 1), 'Loss =', '{:.9f}'.format(c), 'W =', sess.run(W), 'b =', sess.run(b))

    print("Training completed...")

    # Storing necessary values to be used outside the Session
    # https://www.geeksforgeeks.org/linear-regression-using-tensorflow/
    training_cost = sess.run(loss, feed_dict={X: xTrain, Y: yTrain})
    weight = sess.run(W)
    bias = sess.run(b)
    print("Training cost=", training_cost, '; ' "W =", weight, '; ' "b =", bias)

    print("Testing result...")
    test_loss = LinearRegression()  # Same function as above
    testing_cost = sess.run(test_loss, feed_dict={X: xTest, Y: yTest})
    print("Testing cost:", testing_cost)
    print("Absolute mean square loss difference:", abs(training_cost - testing_cost))

    fitted_prediction = sess.run(W) * xTest + sess.run(b)

# Plotting model and results
plt.plot(xTest, fitted_prediction, 'y', label='Fitted line')
plt.legend()
plt.title('Ground truth values vs Prediction')
plt.xlabel('Observations, X')
plt.ylabel('Prediction, Y')
plt.show()
