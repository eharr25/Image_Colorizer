# import libraries
from skimage import io, color
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.neural_network import MLPRegressor
import numpy as np
#Keras test
from keras.models import Sequential
from keras import layers

# import pickle or/and json to save neural network data

# import train and target data
train = np.array(img_to_array(load_img("image_colorizer/Image_colorizer/test0.jpg")), dtype=float)
print(train.shape())
target = np.array(img_to_array(load_img("image_colorizer/Image_colorizer/target0.jpg")), dtype=float)


# convert data from RGB to LAB
x_train = color.rgb2lab(1.0/255*train)[:,:,0] # rgb has 255 values, lab is a percentage. We just want the l layer
x_target = color.rgb2lab(1.0/255*train)[:,:,1:] #Gets layers a and b for the color predictions
x_test = color.rgb2lab(1.0/255*train)[:,:,0]
y_test = color.rgb2lab(1.0/255*train)[:,:,1:]

x_train = x_train.reshape(1,256,256,1)
x_target = x_target.reshape(1,256,256,2)

model = Sequential()
model.add(layers.Dense(256,input_shape=(256,)))
model.add(layers.Dense(128))
model.add(layers.Dense(256))

model.compile(optimizer='rmsprop',loss='mse')

model.fit(x=x_train, y=x_target, batch_size=1, epochs=30)

output = model.predict(x_test)
img = array_to_img(output)
img.show()
