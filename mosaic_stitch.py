from helper import *
import os
import cv2 as cv

if __name__ == '__main__':
    
    dirPath = os.path.join(os.getcwd(), 'imgs')
    mosiac = Mosaic(dirPath)

    drawnImgList = []
    for img, kp in zip(mosiac.imageList, mosiac.kpList):
        drawnImg = cv.drawKeypoints(img, kp, None)
        drawnImgList.append(drawnImg)
    
    DisplayImages(drawnImgList)