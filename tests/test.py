from src.helper import *
import os
import cv2 as cv

def testFeatures():
    drawnImgList = []
    for img, kp in zip(mosiac.imageList, mosiac.kpList):
        drawnImg = cv.drawKeypoints(img, kp, None)
        drawnImgList.append(drawnImg)
    
    DisplayImages(drawnImgList)

if __name__ == '__main__':
    
    dirPath = os.path.join(os.getcwd(), 'imgs')
    mosiac = Mosaic(dirPath)

    # Test feature extraction
    testFeatures()