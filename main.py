import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image as im
from preprocess import preprocess


try: 
    test = cv2.imread("Hindi2.jpg")
    gray_image = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    cv2.imshow('second',gray_image)
    cv2.waitKey(0)
    segments, template, th_img, text_color = preprocess(gray_image)
    # cv2.imshow('thresh',thresh)
    # cv2.waitKey(0)
    cv2.imshow('template',template)
    cv2.waitKey(0)
    # lund=im.fromarray(segments[0])
    # lund.save("boka.jpg")
    for i in range(len(segments)):
        img_cpy=im.fromarray(segments[i])
        img_cpy=img_cpy.resize((32,32))
        img_cpy.show()
        cv2.waitKey(0)


    # img_cpy.save("boka"+str(i)+'.jpg')

except:
    print("Image not found")
