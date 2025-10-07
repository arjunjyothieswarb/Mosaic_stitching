import numpy as np
import cv2 as cv
import gtsam
import math

def Aff2Pose(affineMatrix:np.array) -> gtsam.Pose2:
    theta = math.atan2(affineMatrix[1,0],affineMatrix[0,0]) 
    return gtsam.Pose2(affineMatrix[0,2], affineMatrix[1,2], theta)

def Pose2Aff(pose: gtsam.Pose2) -> np.array:
    return pose.matrix()

def computeNoise(numMatches:int) -> gtsam.noiseModel:
    
    # Setting the coeffs
    coeff_a = 5 #10
    coeff_b = 20 #200
    coeff_c = 10 #100
    
    # Computing the noise
    sigma = coeff_a * (1/(1 + np.exp((numMatches - coeff_b)/coeff_c)))

    return gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma, math.radians(1e-4 * sigma)]))

def StichImagesFromGraph(imageList: list[np.array], HList: list[np.array]):
    
    # Initializing the current min and max
    currMax_X = 0
    currMin_X = float('inf')

    currMax_Y = 0
    currMin_Y = float('inf')

    # Loop through all the transforms to determine the final image size
    for H, image in zip(HList, imageList):
        height, width = image.shape[:2]
        cornerPts = np.float32([ 
            [      0,        0],
            [      0, height-1],
            [width-1, height-1],
            [width-1,        0]
        ]).reshape(-1,1,2) # float32 because perspectiveTransform function takes in float
        
        # Computing the corners of the warped image
        warpedCornerPts = cv.perspectiveTransform(cornerPts, H).reshape(4,2)

        # Getting the max and min values
        max_X, max_Y = np.max(warpedCornerPts, 0)
        min_X, min_Y = np.min(warpedCornerPts, 0)

        currMax_X = np.max((currMax_X, max_X))
        currMax_Y = np.max((currMax_Y, max_Y))

        currMin_X = np.min((currMin_X, min_X))
        currMin_Y = np.min((currMin_Y, min_Y))

    # Get the new height and width of the image
    newHeight = np.int32(currMax_Y - currMin_Y)
    newWidth = np.int32(currMax_X - currMin_X)

    finalImageShape = (newWidth, newHeight)

    # Computing the translation transform to fit the image inside the frame
    translationTransform = np.eye(3)
    translationTransform[0, 2] = -currMin_X
    translationTransform[1, 2] = -currMin_Y

    # Getting the channel size
    if(len(imageList[0].shape))>2:
        channelSize = imageList[0].shape[-1]
    else:
        channelSize = 1
    
    finalImg = np.zeros((newHeight, newWidth, channelSize), dtype=np.uint8)
    for H, image in zip(HList, imageList):
        # Applying the translational transform to all the transforms
        H = translationTransform @ H

        # Translating the in-progress Mosaic image
        warpedImg = cv.warpPerspective(image, H, finalImageShape)

        # Masking the overlapping
        _, mask = cv.threshold(warpedImg, 1, 255, cv.THRESH_BINARY_INV)
        warpedFinalImg = cv.bitwise_and(finalImg, mask)

        # Stitching them together
        finalImg = warpedFinalImg + warpedImg
    
    return finalImg