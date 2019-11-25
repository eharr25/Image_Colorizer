# import libraries
# for image processing
from skimage import io, color
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# neural network
from sklearn.neural_network import MLPRegressor
# so we can save our network
import pickle


#  This might be needed to scale our pictures so they are the same size...
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''


# import test and train data
train1 = load_img("C:/Users/einel/OneDrive/Desktop/BYUI/2019 Fall/Machine Learning/Python code/picture_colorization/train1.jpg")
train2 = load_img("C:/Users/einel/OneDrive/Desktop/BYUI/2019 Fall/Machine Learning/Python code/picture_colorization/train2.jpg")
test1 = load_img("C:/Users/einel/OneDrive/Desktop/BYUI/2019 Fall/Machine Learning/Python code/picture_colorization/test1.jpg")
test2 = load_img("C:/Users/einel/OneDrive/Desktop/BYUI/2019 Fall/Machine Learning/Python code/picture_colorization/test2.jpg")

# convert data from RGB to LAB
x_train = img_to_array(train1)
y_train = img_to_array(train2)
x_test = img_to_array(test1)
y_test = img_to_array(test2)

'''
It might be easier to convert from LAB values to Vector LAB values so that the network doesnt have to deal with a 3d array?
Here I just decided to convert to a d2 array for now
We could also train and run 3 networks, one for each dimension of our 3d array. a newtork for the L, A and B values seperatley
'''
# reshape train vectors so network can read them
nsamples, nx, ny = x_train.shape
x_train = x_train.reshape((nsamples,nx*ny))
nsamples, nx, ny = y_train.shape
y_train = y_train.reshape((nsamples,nx*ny))

# train a basic network
mlp = MLPRegressor(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(x_train,y_train)

# predict passed in images
predictions = mlp.predict(x_test)

# will have to convert back to a 3d array?

# revert to rgb values
back_to_rgb = color.lab2rgb(predictions)

# # export to image
img_pil = array_to_img(back_to_rgb)
img_pil.show()