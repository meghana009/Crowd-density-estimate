import tensorflow as tf
import keras
import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.models import load_model
classifier = tf.keras.models.load_model(r'C:\Users\91992\AppData\Local\Programs\Python\Python37\bus (1).h5')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)



import numpy as np
from keras.preprocessing import image
#import cv2

test_image = image.load_img(r"C:\Users\91992\AppData\Local\Programs\Python\Python37\images\High (715) (2).jpg", target_size = (64,64))
x = image.img_to_array(test_image)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = classifier.predict(images, batch_size=32)
  
  ##cv2.imshow(fn)
  #print(max(classes))
Labels=['High','Low','Moderate','Very High','Very Low']   ### please write your classes names
index=np.argmax(classes)
print(Labels[index])

from firebase import firebase

firebase=firebase.FirebaseApplication(r'https://sampleproj-1dada.firebaseio.com/')

result=firebase.patch('/user',{'new':Labels[index]})
#result=firebase.get('/user/new',None)
print (result)
