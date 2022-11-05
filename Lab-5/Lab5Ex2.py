#!/usr/bin/env python3
"""
Author: Yuvaraj Prem Kumar
Matriculation ID: 303384
Email: premyu@uni-hildesheim.de
Created date: 02.06.2019
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import datasets

def generate_data():
    faces_df=datasets.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)
    xTrain, xTest, yTrain, yTest = train_test_split(faces_df.data, faces_df.target, test_size=0.10, random_state=0)
    nExamples = xTrain.shape[0]
    nFeatures = xTrain.shape[1]
    nLabels = 40
    return xTrain, xTest, yTrain, yTest,nExamples,nFeatures,nLabels


def logistic_regression():
    # https://www.geeksforgeeks.org/ml-logistic-regression-using-tensorflow/
    logits = tf.add(tf.matmul(X, W),b)
    y_pred = tf.nn.softmax(logits)
    prediction = tf.equal(tf.argmax(y_pred, 1), Y)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return logits,accuracy


xTrain, xTest, yTrain, yTest,nExamples,nFeatures,nLabels = generate_data()

# TensorFlow parameters
X = tf.placeholder(tf.float32, [None, nFeatures])
Y = tf.placeholder(tf.int64, [None])  # Int64 used to fit tf.sparse_softmax_cross_entropy

W = tf.get_variable(name='Weight', dtype=tf.float32, shape=([nFeatures, nLabels]), initializer=tf.zeros_initializer())
b = tf.get_variable(name='Bias', dtype=tf.float32, shape=([nLabels]), initializer=tf.zeros_initializer())

logits,accuracy = logistic_regression()
# logits array can be squashed to a valid distribution by a softmax. This is what tf.sparse_softmax_cross_entropy does internally.
# https://stackoverflow.com/questions/49883631/logits-representation-in-tensorflow-s-sparse-softmax-cross-entropy
loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=Y,logits=logits))

# Choose the optimizer and vary learning rate here
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# Initialize variables
training_loss = []
train_accuracy = []
test_accuracy = []

epochs=1000
display_step = 200

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(epochs):
        _, c,corr = sess.run([optimizer, loss, accuracy], feed_dict={X: xTrain, Y: yTrain})  # Training dataset
        # print and append training loss, training accuracy from above function
        training_loss.append(c)
        train_accuracy.append(corr)

        # print and append testing loss, testing accuracy from above function
        _, ctest, corrtest = sess.run([optimizer, loss, accuracy], feed_dict={X: xTest, Y: yTest})
        test_accuracy.append(corrtest)

        # Print output step-wise
        if (e + 1) % display_step == 0:
            print('Epoch #:', '%d' % (e + 1), 'Training Loss =', '{:.9f}'.format(c),'Training accuracy= {:.5f}'.format(corr))

    print("Training completed.....")
    # Final training cost and accuracy
    print('\nFinal training loss: {}',format(c))
    print('Train accuracy: {}'.format(sess.run(accuracy, feed_dict={X: xTrain, Y: yTrain})))
    # Final testing cost and accuracy
    print('\nCost in test set: {}'.format(ctest))
    print('Test accuracy: {}'.format(sess.run(accuracy, feed_dict={X: xTest, Y: yTest})))

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




























