import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

fig = plt.figure()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x1=x_train[7]
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(x1)
plt.title('Class 6: Horse - Original')

datagen = ImageDataGenerator(
    rotation_range=180, # Randomly rotate by degrees
    #width_shift_range=0.2,  # For translating image vertically
    #height_shift_range=0.2, # For translating image horizontally
    #horizontal_flip=True,
    rescale=None,
    fill_mode='nearest',
)
datagen.fit(x_train)

x_batch = datagen.flow(x_train,shuffle=False)[0]
img = x_batch[7] / 255
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img)
plt.title('Class 6: Horse - Random Rotation')

datagen = ImageDataGenerator(
    #rotation_range=180, # Randomly rotate by degrees
    width_shift_range=0.3,  # For translating image vertically
    height_shift_range=0.3, # For translating image horizontally
    #horizontal_flip=True,
    rescale=None,
    fill_mode='nearest',
)#
datagen.fit(x_train)

x_batch = datagen.flow(x_train,shuffle=False)[0]
img = x_batch[7] / 255
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(img)
plt.title('Class 6: Horse - Random Translation')

datagen = ImageDataGenerator(
    #rotation_range=180, # Rotate by degrees
    #width_shift_range=0.4,  # For translating image vertically
    #height_shift_range=0.4, # For translating image horizontally
    #horizontal_flip=True,
    rescale=2, # For rescaling images
    fill_mode='nearest', # Fill empty pixels
)
datagen.fit(x_train)
x_batch = datagen.flow(x_train,shuffle=False)[0]
img = x_batch[7] / 255
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img)
plt.title('Class 6: Horse - Random scaling')
plt.tight_layout()
plt.show()