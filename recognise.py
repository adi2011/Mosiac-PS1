from preprocess import preprocess, detect_text, localize
from prediction import prediction
import numpy as np
import matplotlib.pyplot as plt
import cv2

def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov
# test=input("Enter the address")

def detect_text(main_image, gray_img, localized, bc):        
        cimg = cv2.resize(localized, (30, 30))
        bordersize = 1
        nimg = cv2.copyMakeBorder(cimg, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[255-bc, 0, 0])

        return main_image, nimg

img = cv2.imread('hindi5.jpeg')
img=cv2.resize(img,(350,round(350*(img.shape[0]/img.shape[1]))),interpolation = cv2.INTER_AREA)
print(img.shape)
# scale_percent = 50

# #calculate the 50 percent of original dimensions
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)

# # dsize
# dsize = (width, height)

# # resize image
# img = cv2.resize(img, dsize)
# # img=shadow_remove(img)
# # img = cv2.ximgproc.thinning(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('second',gray_image)
cv2.waitKey(0)
segments, template, th_img, text_color = preprocess(gray_image)
# cv2.imshow('thresh',thresh)
# cv2.waitKey(0)
cv2.imshow('template',template)
cv2.waitKey(0)

labels = []
accuracy = []
show_img = gray_image[:]
#print(len(segments))
    
for segment in segments: 
    plt.imshow(segment)
    plt.show()
    recimg, bimg = detect_text(show_img, th_img, segment, text_color)
        #print('Process: Recognition....\n')
    label, sure = prediction(bimg)
    if(sure > 80):
            #print(segment)
            labels.append(str(label))
            accuracy.append(sure)
            show_img = localize(show_img, th_img, segment, text_color, show)
    char = labels
    accuracy = np.average(accuracy)
    char = ''.join(char)
    if accuracy < 80:
        recimg, bimg = detect_text(show_img, th_img, template, text_color)
        show_img = localize(show_img, th_img, template, text_color, show)
        char, accuracy = prediction(bimg)
        
    plt.imshow(show_img)
    plt.title('Detecting')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    print('The prediction accuracy for ', char,' is ',"%.2f" % round(accuracy,2), '%')
