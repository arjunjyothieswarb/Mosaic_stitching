import numpy as np
import gtsam
import math

def Aff2Pose(affineMatrix:np.array):
    theta = math.atan2(affineMatrix[1,0],affineMatrix[0,0]) 
    return gtsam.Pose2(affineMatrix[0,2], affineMatrix[1,2], theta)

def Pose2Aff(pose: gtsam.Pose2):
    return pose.matrix()