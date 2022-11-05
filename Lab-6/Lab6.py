#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 14.06.2019
"""

import pandas as pd
import tensorflow as tf
print("TensorFlow ver: ",tf.__version__)
import numpy as np

def generate_data(filelist, sliding_step):
    dir = 'C:/PythonProjects/PAMAP2_Dataset/Protocol/'
    # Required columns
    columns = [1,  # Activity ID
               38, 39, 40  # IMU_ankle , 3D-acceleration data for leg part
               ]
    # Required rows
    ID_rows = [3, 4, 12, 13]  # standing, walking, ascending stairs, descending stairs
    data = []

    for file in filelist:
        input = dir + file
        print('Reading data file: ', input)
        df = pd.read_csv(input, header=None, delim_whitespace=True)
        df = df.fillna(df.mean())  # Replace NaNs with mean of column
        df = df[df[1].isin(ID_rows)]  # Keep only the required rows & drop the rest
        df = df[columns]  # Keep only the required columns & drop the rest
        data.append(df)
    df = pd.concat(data)  # Merge into one dataframe

    # Normalize X_dataframe as per http://cs231n.github.io/neural-networks-2/
    x_df = df.drop(1, axis=1)  # Drop target column (Activity ID)
    x_df -= np.mean(x_df, axis=0)
    x_df /= np.std(x_df, axis=0)

    y_df = pd.get_dummies(df[1])  # One-hot encoding for this categorical target data (Activity ID)

    # Sliding window based on research paper
    X_data,Y_data=sliding_window(x_df,y_df,sliding_step)
    return X_data, Y_data

# https://stackoverflow.com/questions/44790072/sliding-window-on-time-series-data
def sliding_window(x_df,y_df,step):
    window_size = 256  # Constant
    ylen = len(y_df)  # Length of target
    max_step = ylen // step  # Sliding over dataframe
    X_data = []
    Y_data = []
    # https://www.geeksforgeeks.org/window-sliding-technique/
    for i in range(0, max_step):
        start = i * step
        end = window_size + (i * step)
        if end > ylen:
            start = ylen - window_size
            end = ylen
        X_data.append(np.array(x_df[start:end]))
        arr = y_df[start:end]
        Y_data.append(np.array(arr.sum() / len(arr)))
    # Convert to ndarray for TensorFlow
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    return X_data, Y_data


train_filelist = ['subject102.dat', 'subject103.dat', 'subject104.dat', 'subject105.dat','subject106.dat','subject107.dat']
test_filelist = ['subject101.dat']

# Calling function to generate training dataset
print('Generating training dataset')
xTrain, yTrain = generate_data(train_filelist, sliding_step=128)
print('Shape of X_Train: ', xTrain.shape)  # (3891, 256, 3)
print('Shape of Y_Train: ', yTrain.shape)  # (3891, 4)

# Calling function to generate test dataset
print('Generating testing dataset')
xTest, yTest = generate_data(test_filelist, sliding_step=128)
print('Shape of X_Test: ',xTest.shape)  # (584, 256, 3)
print('Shape of Y_Test: ',yTest.shape)  # (584, 4)

# From shape of input channels (IMU Ankle)
nWidth = 256
nChannels = 3
nLabels = 4
# Stated in research paper
kernel_size = 5
depth = 8
nHidden = 732

# TensorFlow graph
X = tf.placeholder(tf.float32, [None, nWidth, nChannels])
Y = tf.placeholder(tf.float32, [None, nLabels])

# The code block onwards follows exactly the code demonstration by Mofassir during Thurdsay Lab Session (Group 2)
# The example was with xavier_initializer, but that makes the accuracy jump to 1.0
# Here I'm using all tf.random_normal

weight_decay = tf.constant(0.0005, dtype=tf.float32)  # Weight decay implementation
weights = {
    'wc1': tf.get_variable('W0',shape=([kernel_size,nChannels,depth*nChannels]),initializer=tf.random_normal_initializer,
                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay)),
    'wc2': tf.get_variable('W1',shape=([kernel_size,depth*nChannels,nLabels*nChannels]),initializer=tf.random_normal_initializer,
                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay)),
    'wd1': tf.get_variable('W2',shape=([nHidden,12]),initializer=tf.random_normal_initializer,
                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay)),
    'out': tf.get_variable('W3',shape=([12,nLabels]),initializer=tf.random_normal_initializer,
                           regularizer=tf.contrib.layers.l2_regularizer(weight_decay)),
}

biases = {
    'b1': tf.get_variable('B1',shape=([depth*nChannels]),initializer=tf.random_normal_initializer),
    'b2': tf.get_variable('B2',shape=([nLabels*nChannels]),initializer=tf.random_normal_initializer),
    'b3': tf.get_variable('B3',shape=([12]),initializer=tf.random_normal_initializer),
    'out': tf.get_variable('B4',shape=([nLabels]),initializer=tf.random_normal_initializer),
}


def conv1d(X,feature_maps,bias):
    conv = tf.nn.conv1d(X,feature_maps,stride=1,padding="VALID")  # Stride=1 because 1-D, no need zero padding
    conv = tf.nn.sigmoid(tf.nn.bias_add(conv, bias)) # Paper is using sigmoid activation function
    return conv


# Paper states average pooling used instead of max pooling
def avgpool1d(conv):
    pooling = tf.layers.average_pooling1d(conv, pool_size=2, strides=2, padding='VALID')  # no need zero padding
    return pooling


# MC-DCNN implementation
def conv_net(X,weights,biases):
    conv1 = conv1d(X,weights['wc1'],biases['b1'])  # First layer, feature maps of 252 per input channel
    pool1 = avgpool1d(conv1)  # 1-D average pooling, chooses avg. value in the matrix

    conv2 = conv1d(pool1,weights['wc2'],biases['b2'])  # Second layer, feature map of 126 per input channel
    pool2 = avgpool1d(conv2)  # Avg pooling again

    # Fully connected layer
    cnn = tf.layers.Flatten()(pool2)  # Reshape conv2 pooling output to fit fully connected layer input
    fc1 = tf.add(tf.matmul(cnn, weights['wd1']), biases['b3'])
    fc1 = tf.nn.sigmoid(fc1)  # Paper is using sigmoid activation function

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out = tf.nn.softmax(out)  # Final output layer of 4 classes (corresponding to activity ID)
    return out


learning_rate=0.05
training_epochs=20
batch_size=128


# The code block onwards follows exactly the code demonstration by Mofassir during Thurdsay Lab Session (Group 2)
# CONV_2D was presented, but the same logic is applied to my CONV_1D
pred=conv_net(X,weights,biases)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

total_batches = xTrain.shape[0] // batch_size

# Tensorboard
tf.summary.histogram("Activations", pred)
tf.summary.scalar("Loss_on_train_set", loss)
tf.summary.scalar("Accuracy_on_train_set", accuracy)
merged_all_ = tf.summary.merge_all()

# Initialize
training_loss = []
train_accuracy = []
test_accuracy = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    summary_writer = tf.summary.FileWriter('./tensorboard/train', graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        for batch in range(total_batches):  # Mini-batch for SGD
            batch_x = xTrain[batch * batch_size:min((batch + 1) * batch_size, len(xTrain))]
            batch_y = yTrain[batch * batch_size:min((batch + 1) * batch_size, len(yTrain))]
            _, c, acc,summary = sess.run([optimizer, loss,accuracy, merged_all_], feed_dict={X: batch_x, Y: batch_y})
            training_loss.append(c)
            summary_writer.add_summary(summary, epoch)
        print('Epoch: ', epoch, 'Training Loss =', '{:.9f}'.format(c), 'Training accuracy= {:.5f}'.format(acc))
        train_accuracy.append(acc)
        _, corrtest = sess.run([optimizer, accuracy], feed_dict={X: xTest, Y: yTest})
        test_accuracy.append(corrtest)

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: xTest, Y: yTest}))

# Plotting code is from my Lab 5 submission
from matplotlib import pyplot as plt

plt.plot(training_loss)
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.show()

plt.plot(train_accuracy, 'r-', label='train accuracy')
plt.plot(test_accuracy, 'g-', label='test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train / Test Accuracy')
plt.legend()
plt.show()