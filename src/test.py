from utils import *
import os
import cv2 as cv

def testFeatures(mosaic: Mosaic)->None:
    drawnImgList = []
    mosaic.kpList, mosaic.desList = mosaic.extractFeatures()
    for img, kp in zip(mosiac.imageList, mosiac.kpList):
        drawnImg = cv.drawKeypoints(img, kp, None)
        drawnImgList.append(drawnImg)
    
    DisplayImages(drawnImgList)

if __name__ == '__main__':
    
    dirPath = os.path.join(os.getcwd(), 'imgs')
    mosiac = Mosaic(dirPath)

    # Loading images
    mosiac.imageList = LoadImages(mosiac.dirPath, (mosiac.scaleDownFactor, mosiac.scaleDownFactor))
    mosiac.imageCount = len(mosiac.imageList) # Updating imageCount

    # Test feature extraction
    testFeatures(mosiac)