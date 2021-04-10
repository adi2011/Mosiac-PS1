from preprocess import preprocess, detect_text, localize
import numpy as np
import matplotlib.pyplot as plt
import cv2
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
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

image1 = cv2.imread('hindi3.jpg')
img1=shadow_remove(image1)
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_image=image_resize(img, width=1200)
cv2.imshow('second',gray_image)
cv2.waitKey(0)
segments, template, th_img, text_color = preprocess(gray_image)
# cv2.imshow('thresh',thresh)
# cv2.waitKey(0)
cv2.imshow('th_img',th_img)
cv2.waitKey(0)

labels = []
accuracy = []
show_img = gray_image[:]
#print(len(segments))
    
for segment in segments: 
    plt.imshow(segment)
    plt.show()
    recimg, bimg = detect_text(show_img, th_img, segment, text_color)