import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image as im
import numpy as np
from predictor import prediction
from keras.preprocessing import image
from preprocess import preprocess
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2


dest=input("Enter file location and hit enter after every image ")
 
test = cv2.imread(dest)
gray_image = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
cv2.imshow('second',gray_image)
cv2.waitKey(0)
segments, template, th_img, text_color = preprocess(gray_image)
    # cv2.imshow('thresh',thresh)
    # cv2.waitKey(0)
cv2.imshow('template',template)
cv2.waitKey(0)
    
    # labels = '०,१,२,३,४,५,६,७,८,९,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ'
    # labels = labels.split(',')
    # print(labels[2])
for i in range(len(segments)):
    img_cpy=cv2.resize(segments[i],(32,32))
    img_cpy = np.array(img_cpy,dtype="float32")
    img_cpy = np.expand_dims(img_cpy, axis=-1)
    img_cpy/=255.0
    # print (img_cpy)
    cv2.imshow('template',img_cpy)
    letter,succ,out=prediction(img_cpy)
    print(out)
    cv2.waitKey(0)
        


# img_cpy=im.fromarray(segments[i])
#         img_cpy=img_cpy.resize((32,32))
#         letter=np.array(img_cpy)
#         letter=letter.astype(numpy.float32)
#         
# img_cpy.show()
# 

