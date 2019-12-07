# import libraries
import tensorflow as backend
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, InputLayer, UpSampling2D
import os

import pickle


def get_lab(img):
    l = rgb2lab(img/255)[:,:,0]
    return l

def get_color(img):
    x = rgb2lab(img/255)[:,:,1:] # this is the A and B values; a-magenta-green; b-yellow-blue
    x/=128
    return x

def get_images(path, color="lab"):
    images = list()
    for filename in os.listdir(path):
        if filename[0] != '.':
            if color == "lab":
                images.append(get_lab(np.array(img_to_array(load_img(path + filename)), dtype=float)))
            else:
                images.append(get_color(np.array(img_to_array(load_img(path + filename)), dtype=float)))
    return images

#print (get_images("./TrainImages/"))


'''
Convert all training images from the RGB color space to the Lab color space.
Use the L channel as the input to the network and train the network to predict the ab channels.
Combine the input L channel with the predicted ab channels.
Convert the Lab image back to RGB.
'''

# The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
# (samples, rows, cols, channels), so we need to do some reshaping
# https://keras.io/layers/convolutional/
x = get_images("./TrainImages/") #l value only
y = get_images("./TrainImages/", color="yes") #a and b values
    
x = np.array(x)
print("X shape: {}".format(x.shape))
y = np.array(y)
x = x.reshape(x.shape[0], x.shape[1], x.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], y.shape[1], 2)

# create model
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1))) # input shape is only needed for first layer? input_shape=(256, 256, 3)
# 3x3 kernel used and 8 filters?
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
# figure out what this does
# model.add(layers.MaxPooling2D((2, 2)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))
# get working after we get NN working better
'''
# supposed to soften image
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
'''
# get summary of layers and compile
model.summary()
model.compile(optimizer='adam',loss='mse') # loss='sparse_categorical_crossentropy', optomizer='rmsprop'
model.fit(x=x,y=y, batch_size=10,verbose=0, epochs=100)

# evaluate model
model.evaluate(x, y, batch_size=10)

# convert to lab- black and white image
z=get_lab(np.array(img_to_array(load_img("./TestImages/test0.jpg")), dtype=float))
z = z.reshape(1, z.shape[0], z.shape[1], 1)

# make predictions
output = model.predict(z)
print(output.shape)
output*=128

# make sure output has the correct shape
cur = np.zeros(z.shape)
cur[:,:,0] = z[0][:,:,0] # L layer?
cur[:,:,1:] = output[0] # A B layers?
print(cur.shape)

# convert to rgb
rgb_image = lab2rgb(cur)

img = array_to_img(rgb_image)
img.show()