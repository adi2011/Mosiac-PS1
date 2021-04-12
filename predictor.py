import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image as im
from keras.preprocessing import image
from preprocess import preprocess
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2


def prediction(img):
    # load json and create model
    # json_file = open('cnn2\cnn2.json', 'r')
    
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    model = tf.keras.models.load_model("model.h5py")
    # model=Sequential()
    # model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,1)))
    # model.add(AveragePooling2D())
    # model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    # model.add(AveragePooling2D())
    # model.add(Flatten())
    # model.add(Dense(units=120, activation='relu'))
    # model.add(Dense(units=84, activation='relu'))
    # model.add(Dense(units=26, activation = 'softmax'))
    # model.note_weights('weight.h5')
    # model.load_weights("weight.h5")
    #print("Loaded model from disk")
    
    # loaded_model.save('cnn.hdf5')
    # loaded_model=load_model('cnn.hdf5')
    # lists = model.predict(img_cpy)
    # characters = '०,१,२,३,४,५,६,७,८,९,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ'
    # characters = characters.split(',')
    
    x = np.asarray(img, dtype = np.float32).reshape(1,32, 32,1) / 255 
    output = model.predict(x)[0]
    output = output.reshape(26)
    predicted = np.argmax(output)
    success = output[predicted] * 100
    
    return predicted, success
