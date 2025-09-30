from utils import *
import os
import cv2 as cv

def testFeatures(mosaic: Mosaic, displayFlag: bool=True)->None:
    drawnImgList = []
    mosaic.kpList, mosaic.desList = mosaic.extractFeatures()
    for img, kp in zip(mosaic.imageList, mosaic.kpList):
        drawnImg = cv.drawKeypoints(img, kp, None)
        drawnImgList.append(drawnImg)
    
    # Display the image if flag is set
    if displayFlag:
        DisplayImages(drawnImgList, (mosaic.scaleDownFactor, mosaic.scaleDownFactor))
    
    return None

def testFeatureMatch(mosaic: Mosaic, displayFlag: bool=True)->None:
    
    # Finding matches
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
    
    # Display the image if flag is set
    if displayFlag:
        DisplayImages(drawnImgList, (mosaic.scaleDownFactor, mosaic.scaleDownFactor))
    
    return None

def testRANSAC(mosaic: Mosaic, displayFlag: bool=True)->None:
    
    # Computing the homography between images and getting the masks
    mosaic.HomographyTransforms, mosaic.matchesMaskList = mosaic.computeHomography()
    
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
        matchesMask = mosaic.matchesMaskList[idx]

        # Masking to get only the inliers
        good = [matches[i] for i in range(len(matches)) if matchesMask[i]==1]
        
        # Drawn image
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        drawnImgList.append(img3)
    
    # Display the image if flag is set
    if displayFlag:
        DisplayImages(drawnImgList, (mosaic.scaleDownFactor, mosaic.scaleDownFactor))
    
    return None

def testStitching(mosaic: Mosaic, scale: tuple=None)->None:
    finalImg = mosaic.stitchImages(mosaic.imageList, mosaic.HomographyTransforms)
    
    DisplayImages([finalImg], (mosaic.scaleDownFactor, mosaic.scaleDownFactor))
    return None

if __name__ == '__main__':
    
    # dirPath = os.path.join(os.getcwd(), 'imgs')
    with open("./config/config.yaml") as f:
            config = yaml.safe_load(f)

    mosaic = Mosaic(config)
    mosaic.scaleDownFactor = 0.7

    # Loading images
    mosaic.imageList = LoadImages(mosaic.dirPath)
    mosaic.imageCount = len(mosaic.imageList) # Updating imageCount

    # Preprocessing
    blurredImgList = []
    kernelSize = 3
    for img in mosaic.imageList:
        blurredImg = cv.GaussianBlur(img, (kernelSize, kernelSize), 0)
        blurredImg = cv.normalize(blurredImg, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        blurredImgList.append(blurredImg)
    
    # Overwriting the list with processed images
    mosaic.imageList = blurredImgList

    # Test feature extraction
    testFeatures(mosaic, False)

    # Test feature matching
    testFeatureMatch(mosaic, False)

    # Test RANSAC filtering
    testRANSAC(mosaic, False)

    # Test stitching
    testStitching(mosaic)
