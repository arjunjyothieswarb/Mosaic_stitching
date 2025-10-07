import os
import math
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

import gtsam
import gtsam.utils.plot as gtsam_plot

from utils import *
from graph_utils import *

with open("./config/config.yaml") as f:
        config = yaml.safe_load(f)

def PreProcessImages(mosaic: Mosaic):
    # Preprocessing
    blurredImgList = []
    kernelSize = config["Preprocessing"]["GaussianBlur"]["kernelSize"]
    clahe = cv.createCLAHE(clipLimit=5)
    for img in mosaic.grayList:
        blurredImg = cv.GaussianBlur(img, (kernelSize, kernelSize), 0)
        # blurredImg = cv.normalize(blurredImg, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        blurredImg = clahe.apply(blurredImg)
        blurredImgList.append(blurredImg)
    
    return blurredImgList

if __name__ == '__main__':

    # Initializing Mosiac object
    mosaic = Mosaic(config)

    # Loading images
    mosaic.imageList, mosaic.grayList = LoadImages(mosaic.dirPath)
    mosaic.imageCount = len(mosaic.imageList) # Updating imageCount
    
    # Overwriting the list with processed images
    mosaic.grayList = PreProcessImages(mosaic)

    # Extracting features
    mosaic.kpList, mosaic.desList = mosaic.extractFeatures(mosaic.grayList)

    # Anchoring the first image
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-3)

    # Initializing the graph
    graph = gtsam.NonlinearFactorGraph()
    inital_estimate = gtsam.Values()

    # Adding the first image to graph with prior noise
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), priorNoise))
    inital_estimate.insert(0, gtsam.Pose2(0, 0, 0))
    

    # Creating the graph
    H_TransformList = []
    H_actual = np.eye(3)
    for i in range(len(mosaic.grayList) - 1):
        j = i + 1
        while(j < len(mosaic.grayList)):
            # Computing the Affine transform
            H, numMatches = mosaic.computeAffine(i, j)

            # Checking for sequential images
            if j-i == 1:
                if numMatches == -1:
                    print("[ERROR]: Not enough matches found between sequential images {} and {}! - {}/{}".format(i, j, numMatches, config["FeatureMatching"]["MIN_MATCH_COUNT"]))
                    print("Exiting...")
                
                H_actual = H @ H_actual
                inital_estimate.insert(j, Aff2Pose(H_actual))
            
            # If less matches found, ignore
            if numMatches == -1:
                 continue
            
            # Computing pose and noise
            pose = Aff2Pose(H)
            odomNoise = computeNoise(numMatches)

            graph.add(gtsam.BetweenFactorPose2(i, j, pose, odomNoise))

            # Updating the iterator
            j = j + 1
    
    params = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(graph, inital_estimate, params)

    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, inital_estimate)

    for i in range(len(mosaic.grayList)):
        gtsam_plot.plot_pose2(0, inital_estimate.atPose2(i), 10)
        gtsam_plot.plot_pose2(1, result.atPose2(i), 10, marginals.marginalCovariance(i))
    
    plt.show()
    
    
    
    # print("\nFactor Graph:\n{}".format(graph))

    # finalImage = mosaic.stitchImages(mosaic.imageList, H_TransformList)

    # DisplayImages([finalImage], (mosaic.scaleDownFactor, mosaic.scaleDownFactor))