import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# From readme.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# https://github.com/deep-diver/CIFAR10-img-classification-tensorflow/blob/master/CIFAR10_image_classification.ipynb
def onehotencoding(labels, nclasses):
    outlabels = np.zeros((len(labels), nclasses))
    for i, l in enumerate(labels):
        outlabels[i, l] = 1
    return outlabels


# https://github.com/deep-diver/CIFAR10-img-classification-tensorflow/blob/master/CIFAR10_image_classification.ipynb
def normalize_data(x):  # Min-max normalization of pixel values (to [0-1] range)
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x



dir = 'C:/PythonProjects/cifar-10-batches-py/'
data_train = unpickle(dir+'data_batch_1')  # Only can read one batchfile, due to RAM limitations
data_test = unpickle(dir+'test_batch')

def generate_data(input_data,range):
    X_data = input_data[b'data'] # m * n
    X_data = X_data[0:range]
    X_data = normalize_data(X_data)
    X_data = X_data.reshape(-1, 32, 32, 3)

    Y_data = np.array(input_data[b'labels'])
    Y_data = Y_data[0:range]
    Y_data = onehotencoding(Y_data,10) # 10 image classes

    return X_data,Y_data


xTrain, yTrain = generate_data(data_train,7000)  # Call function to generate training X and Y
print('Shape of X_Train data: ',xTrain.shape)
print('Shape of Y_Train labels: ',yTrain.shape)

xTest, yTest = generate_data(data_test,3000)  # Call function to generate testing X and Y
print('Shape of X_Test data: ',xTest.shape)
print('Shape of Y_Test labels: ',yTest.shape)


datagen = ImageDataGenerator(
    rotation_range=180, # Rotate by degrees
    width_shift_range=0.4,  # For translating image vertically
    height_shift_range=0.4, # For translating image horizontally
    horizontal_flip=True,
    rescale=2, # For rescaling images
    fill_mode='nearest', # Fill empty pixels
)
datagen.fit(xTrain)


nInput = 32
nChannels = 3
nClasses = 10

# Placeholder and drop-out
X = tf.placeholder(tf.float32, [None, nInput, nInput, nChannels])
Y = tf.placeholder(tf.float32, [None, nClasses])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.selu(x)  # Using self-normalization activation


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def normalize_layer(pooling):
    #norm = tf.nn.lrn(pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    norm = tf.contrib.layers.batch_norm(pooling, center=True, scale=True)
    return norm


def drop_out(fc, keep_prob=0.4):
    drop_out = tf.layers.dropout(fc, rate=keep_prob)
    return drop_out


weights = {
    'WC1': tf.Variable(tf.random_normal([5, 5, 3, 32]), name='W0'),
    'WC2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='W1'),
    'WD1': tf.Variable(tf.random_normal([8 * 8 * 64, 64]), name='W2'),
    'WD2': tf.Variable(tf.random_normal([64, 128]), name='W3'),
    'out': tf.Variable(tf.random_normal([128, nClasses]), name='W5')
}

biases = {
    'BC1': tf.Variable(tf.random_normal([32]), name='B0'),
    'BC2': tf.Variable(tf.random_normal([64]), name='B1'),
    'BD1': tf.Variable(tf.random_normal([64]), name='B2'),
    'BD2': tf.Variable(tf.random_normal([128]), name='B3'),
    'out': tf.Variable(tf.random_normal([nClasses]), name='B5')
}

def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['WC1'], biases['BC1'])
    conv1 = maxpool2d(conv1)
    conv1 = normalize_layer(conv1)

    conv2 = conv2d(conv1, weights['WC2'], biases['BC2'])
    conv2 = maxpool2d(conv2)

    fc1 = tf.reshape(conv2, [-1, weights['WD1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['WD1']), biases['BD1'])
    fc1 = tf.nn.selu(fc1)  # Using self-normalization activation
    fc1 = drop_out(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['WD2']), biases['BD2'])
    fc2 = tf.nn.selu(fc2)  # Using self-normalization activation
    fc2 = drop_out(fc2)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    out = tf.nn.softmax(out)

    return out




# Hyperparameters
training_epochs = 10
learning_rate = 0.05
batch_size = 10
total_batches = xTrain.shape[0] // batch_size

pred = conv_net(X, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
# Tried 3 different optimizer, seems RMSProp is the best (of the worst for my code)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Tensorboard
tf.summary.histogram("Activations", pred)
tf.summary.scalar("Loss_on_train_set", loss)
tf.summary.scalar("Accuracy_on_train_set", accuracy)
merged_all_ = tf.summary.merge_all()

# The rest of the code is exactly lifted from my Lab 6 submission
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
        print('Epoch: ', epoch, 'Training Loss =', '{:.9f}'.format(c), 'Training accuracy = {:.5f}'.format(acc))
        train_accuracy.append(acc)
        _, corrtest = sess.run([optimizer, accuracy], feed_dict={X: xTest, Y: yTest})
        test_accuracy.append(corrtest)

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: xTest, Y: yTest}))

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