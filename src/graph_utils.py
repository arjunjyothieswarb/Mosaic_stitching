import numpy as np
import gtsam
import math

def Aff2Pose(affineMatrix:np.array) -> gtsam.Pose2:
    theta = math.atan2(affineMatrix[1,0],affineMatrix[0,0]) 
    return gtsam.Pose2(affineMatrix[0,2], affineMatrix[1,2], theta)

def Pose2Aff(pose: gtsam.Pose2) -> np.array:
    return pose.matrix()

def computeNoise(numMatches:int) -> gtsam.noiseModel:
    
    # Setting the coeffs
    coeff_a = 10
    coeff_b = 200
    coeff_c = 100
    
    # Computing the noise
    sigma = coeff_a * (1/(1 + np.exp((numMatches - coeff_b)/coeff_c)))

    # return gtsam.noiseModel.Diagonal.Sigmas(np.array([100, 100, math.radians(1)]))
    return gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma, math.radians(0.1 * sigma)]))