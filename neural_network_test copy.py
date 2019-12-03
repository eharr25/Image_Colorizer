# import libraries
from skimage import io, color
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.neural_network import MLPRegressor
import numpy as np
#Keras test
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, InputLayer
#Color imports
# import pickle or/and json to save neural network data


#  This might be needed to scale our pictures so they are the same size...
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''

# example of converting an image with the Keras API
'''
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

# load the image
img = load_img('bondi_beach.jpg')
print(type(img))

# convert to numpy array # I hope these are the RGB values here
img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)

# convert back to image
img_pil = array_to_img(img_array)
print(type(img))
'''


# import test and train data
train1 = img_to_array(load_img("test0.jpg"))
train1 = np.array(train1, dtype=float)
train2 = img_to_array(load_img("test0.jpg"))
train2 = np.array(train2, dtype=float)
test1 = img_to_array(load_img("test0.jpg"))
test1 = np.array(test1, dtype=float)
test2 = img_to_array(load_img("test0.jpg"))
test2 = np.array(test2, dtype=float)

#print(np.shape(x_target))
# convert data from RGB to LAB
x_train = color.rgb2lab(1.0/255*train1)[:,:,0] # rgb has 255 values, lab is a percentage. We just want the l layer
x_target = color.rgb2lab(1.0/255*train1)[:,:,1:] #Gets layers a and b for the color predictions

x_target = x_target / 128

print(x_target[0,0,0])

print(np.shape(x_train))
print(np.shape(x_target))
x_train = x_train.reshape(1,256,256,1)
x_target = x_target.reshape(1,256,256,2)

'''
It might be easier to convert from LAB values to Vector LAB values so that the network doesnt have to deal with a 3d array?
Here I just decided to convert to a d2 array for now
We could also train and run 3 networks, one for each dimension of our 3d array. a newtork for the L, A and B values seperatley
'''
# reshape train vectors so network can read them
#nsamples, nx, ny = x_train.shape
#x_train = x_train.reshape((nsamples,nx*ny))
#nsamples, nx, ny = x_target.shape
#x_target = x_target.reshape((nsamples,nx*ny))

# train a basic network
#mlp = MLPRegressor(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
#mlp.fit(x_train,x_target)

#print(np.shape(x_train))
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same', strides=2))
model.summary()
#model.add(layers.Dense(256,input_shape=(1,256,256)))
#model.add(layers.Dense(128))
#model.add(layers.Dense(256))

model.compile(optimizer='rmsprop',loss='mse')

model.fit(x=x_train, y=x_target, batch_size=1, epochs=30)

output = model.predict(x_test)
img = array_to_img(output)
img.show()

# predict passed in images
#predictions = mlp.predict(x_test)

# will have to convert back to a 3d array?

# revert to rgb values
#back_to_rgb = color.lab2rgb(predictions)

# # export to image
#img_pil = array_to_img(back_to_rgb)
#img_pil.show()