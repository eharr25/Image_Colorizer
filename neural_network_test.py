# import libraries
# for pre-processing
from sklearn.model_selection import train_test_split
# for image processing
from skimage import io, color
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# neural network
from sklearn.neural_network import MLPRegressor
# so we can save our network
import pickle





# import test and train data
print("Loading data")
train0 = load_img("picture_colorization/test0.jpg")
# train1 = load_img("picture_colorization/train2.jpg")
target0 = load_img("picture_colorization/target0.jpg")
# target1 = load_img("picture_colorization/test2.jpg")

# convert data from RGB to LAB
print("Converting to LAB values")
train_data = img_to_array(train0)
# y_train = img_to_array(train2)
target_data = img_to_array(target0)
# y_test = img_to_array(test2)

#  This might be needed to scale our pictures so they are the same size...
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

It might be easier to convert from LAB values to Vector LAB values so that the network doesnt have to deal with a 3d array?
Here I just decided to convert to a d2 array for now
We could also train and run 3 networks, one for each dimension of our 3d array. a newtork for the L, A and B values seperatley
'''

# train test split
print("Splitting data")
x_train, x_test, y_train, y_test = train_test_split(train_data, target_data, test_size = 20, random_state = 50)


# train a basic network
print("Training network")
mlp = MLPRegressor(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=5000)
mlp.fit(x_train,y_train)


# predict passed in images
predictions = mlp.predict(x_test)

# will have to convert back to a 3d array


# revert to rgb values
print("Reverting to RGB values")
back_to_rgb = color.lab2rgb(predictions)

# # export array to image
print("Exporting image")
img_pil = array_to_img(back_to_rgb)
img_pil.show()