from utils import *
import os
import cv2 as cv



if __name__ == '__main__':
    
    # dirPath = os.path.join(os.getcwd(), 'imgs')
    with open("./config/config.yaml") as f:
            config = yaml.safe_load(f)

    mosaic = Mosaic(config)
    mosaic.scaleDownFactor = config["ScaleFactor"]

    # Loading images
    mosaic.imageList,  mosaic.grayList = LoadImages(mosaic.dirPath)
    mosaic.imageCount = len(mosaic.imageList) # Updating imageCount

    # Preprocessing
    blurredImgList = []
    kernelSize = 3
    for img in mosaic.grayList:
        blurredImg = cv.GaussianBlur(img, (kernelSize, kernelSize), 0)
        blurredImg = cv.normalize(blurredImg, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        blurredImgList.append(blurredImg)
    
    # Overwriting the list with processed images
    # mosaic.grayList = blurredImgList

    # Extracting features
    mosaic.kpList, mosaic.desList = mosaic.extractFeatures(mosaic.grayList)

    # Computing the Homographic transforms
    H_TransformList = []
    for i in range(len(mosaic.grayList) - 1):
        H = mosaic.computeHomography(i, i+1)
        H_TransformList.append(H)
    
    finalImage = mosaic.stitchImages(mosaic.imageList, H_TransformList)

    DisplayImages([finalImage], (mosaic.scaleDownFactor, mosaic.scaleDownFactor))