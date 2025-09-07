from utils import *
import os
import cv2 as cv

def testFeatures(mosaic: Mosaic, displayFlag: bool=True)->None:
    drawnImgList = []
    mosaic.kpList, mosaic.desList = mosaic.extractFeatures()
    for img, kp in zip(mosaic.imageList, mosaic.kpList):
        drawnImg = cv.drawKeypoints(img, kp, None)
        drawnImgList.append(drawnImg)
    
    if displayFlag:
        DisplayImages(drawnImgList)
    
    return None

def testFeatureMatch(mosaic: Mosaic, displayFlag: bool=True)->None:
    mosaic.findMatches()
    
    drawnImgList = []
    for idx in range(mosaic.imageCount - 1):
        
        # Images
        img1 = mosaic.imageList[idx]
        img2 = mosaic.imageList[idx+1]

        # Descriptors
        kp1 = mosaic.kpList[idx]
        kp2 = mosaic.kpList[idx+1]

        # Matches
        matches = mosaic.matchesList[idx]

        # Drawn image
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        drawnImgList.append(img3)
    
    if displayFlag:
        DisplayImages(drawnImgList)
    
    return None

def testRANSAC(mosaic: Mosaic, displayFlag: bool=True)->None:
    
    mosaic.homographyTransforms, mosaic.matchesMask = mosaic.computeHomography()

    for idx in range(mosaic.imageCount - 1):

        # Images
        img1 = mosaic.imageList[idx]
        img2 = mosaic.imageList[idx+1]

        # Descriptors
        kp1 = mosaic.kpList[idx]
        kp2 = mosaic.kpList[idx+1]

        # Matches
        matches = mosaic.matchesList[idx]

        # DrawParams
        drawParams = dict(matchColor = (0,255,0), # Draw matches in green
                          singlePointColor = None,
                          matchesMask = mosaic.matchesMask[idx], # Draw only inliers
                          flags=2)

        # maskedMatches = [matches[i] for i in range(len(matches)) if mosaic.matchesMask[0][i] == 1]
        # print(maskedMatches)
        # Draw image
        drawnImgList = cv.drawMatches(img1, kp1, img2, kp2, matches, None, cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if displayFlag:
        DisplayImages(drawnImgList)
    
    return None

if __name__ == '__main__':
    
    dirPath = os.path.join(os.getcwd(), 'imgs')
    mosaic = Mosaic(dirPath)

    # Loading images
    mosaic.imageList = LoadImages(mosaic.dirPath, (mosaic.scaleDownFactor, mosaic.scaleDownFactor))
    mosaic.imageCount = len(mosaic.imageList) # Updating imageCount

    # Preprocessing
    blurredImgList = []
    kernelSize = 3
    for img in mosaic.imageList:
        blurredImg = cv.GaussianBlur(img, (kernelSize, kernelSize), 0)
        blurredImgList.append(blurredImg)
    
    # Overwriting the list with processed images
    mosaic.imageList = blurredImgList

    # Test feature extraction
    testFeatures(mosaic, False)

    # Test feature matching
    testFeatureMatch(mosaic, False)

    # Test RANSAC filtering
    testRANSAC(mosaic)

