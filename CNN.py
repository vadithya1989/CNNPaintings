#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimage

# Specify image location
DATADIR = "/Users/adithyavijaykumar/Documents/projects/AI/oneShotSiamese/"
CATEGORIES = ["cezanne", "monet", "picasso", "vangogh"]

# Define the training data set, this has both X and y encoded
training_data = []

# Function to create a training data set from the images
def createTrainingData():
#     Loop through the categories
    for category in CATEGORIES:
#         Path to the specific category
        path = os.path.join(DATADIR, category) 
#        Note the category number
        class_num = CATEGORIES.index(category)
#         Loop through all images in a specified category
        for img in sorted(os.listdir(path)):
            try:
#               Read the image
                img_array = cv2.imread(os.path.join(path,img))
#                 img_array = mpimage.imread(os.path.join(path,img))
                IMG_SIZE = 224
#               Resize the image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#               Join the image array with the category number to form trainind data set
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
        
createTrainingData()


# In[2]:


# Shuffle the training data
import random
random.shuffle(training_data)


# In[3]:


# Convert training data from list to nparray
X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)


# In[4]:


IMG_SIZE = 224
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y).reshape(-1, 1)


# In[5]:


# Store X and y so that it can be easily loaded
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0


# In[13]:


model = Sequential()

model.add(Conv2D(16, (3,3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16, (3,3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dense(4, activation='softmax'))


# In[14]:


model.summary()


# In[15]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# In[30]:


model.fit(x_train, y_train, epochs=5, validation_split=0.1)


# In[31]:


model.evaluate(x_test, y_test)


