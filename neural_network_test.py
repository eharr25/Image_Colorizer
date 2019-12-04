# import libraries
from skimage import io, color
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.neural_network import MLPRegressor
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, InputLayer

import pickle



# import train and target data
train = np.array(img_to_array(load_img("image_colorizer/Image_colorizer/test0.jpg")), dtype=float)
# print(train.shape)---(256, 256, 3)
target = np.array(img_to_array(load_img("image_colorizer/Image_colorizer/target0.jpg")), dtype=float)
# print(target.shape)---(256, 256, 3)

# convert data from RGB to LAB and normalize-(we normalize by diving by 255--this gives us a value between 0 and 1)
x_train = color.rgb2lab(1.0/255*train)[:,:,0] # this is the L layer
x_target = color.rgb2lab(1.0/255*train)[:,:,1:] # this is the A and B values

# reshape
# x_train = x_train.reshape(1,256,256,1)
# x_target = x_target.reshape(1,256,256,2)

# create model
model = Sequential()
# need to figure out what to change the layers to and imput shape to 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3))) # input shape is only needed for first layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# should flatten our image to 2d
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.fit(x_train, x_target, epochs=10)




# model.compile(optimizer='rmsprop',loss='mse')

# model.fit(x=x_train, y=x_target, batch_size=1, epochs=30)

# output = model.predict(x_test)
# img = array_to_img(output)
# img.show()