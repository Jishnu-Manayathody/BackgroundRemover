import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

# to open webcam
cap = cv2.VideoCapture(0)
# setting width(3) and height(4)
cap.set(3,640)   # width 640
cap.set(4,480)   # height 480
segmentor = SelfiSegmentation()

# To make a list of background images
listImg = os.listdir("Images")
imgList = []
i = 0
j = 0
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)

while True:
    success, img = cap.read()
    # To remove background and replace it with image (color code like (255,0,0) can be used instead)
    img_out = segmentor.removeBG(img, imgList[i%(len(imgList))], threshold=0.3)
    # Apply Gaussian Blur to blur image
    img_blur = cv2.GaussianBlur(img_out, (21, 21), 0)
    img_blur_list = [img_out,img_blur]
    # To display original and background changed video in single stack
    imgStacked = cvzone.stackImages([img, img_blur_list[(j%2)]], 2, 1)
    cv2.imshow("Image Stacked", imgStacked)

    key = cv2.waitKey(1)
    if key == ord("n"):
        i += 1
    elif key == ord("q"):
        break
    elif key == ord("b"):
        j += 1








