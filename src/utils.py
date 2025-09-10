import cv2 as cv
import numpy as np
import yaml
import os

def LoadImages(dirPath: str, N: float=np.inf) -> list[np.ndarray]:
    """
    Loads all image files from the specified directory and returns them as a list of numpy arrays.
    Args:
        dirPath (str): The path to the directory containing image files.
        scale (tuple, optional): A tuple (fx, fy) specifying the scaling factors for resizing images. Defaults to None.
    Returns:
        list[np.ndarray]: A list of images loaded as numpy arrays.
    Notes:
        - Supported image formats are .tif, .jpg, and .png.
        - Images are read in sorted order by filename.
        - Requires the 'os' and 'cv2' (as 'cv') modules to be imported.
    """

    imageList = []
    
    # Getting all the image files in the directory
    fileNames = [file for file in os.listdir(dirPath) if file.endswith((".tif", ".jpg", ".png"))]
    
    # Reading all the images in order
    count = 0
    for file in sorted(fileNames):
        filePath = os.path.join(dirPath, file)
        image = cv.imread(filePath, cv.IMREAD_GRAYSCALE)

        if image is not None:
            count = count + 1
            if count > N:
                break
            imageList.append(image)
        else:
            print(f"Unable to open {file}! Ignoring {file}")
            continue

    return imageList

def DisplayImages(imageList: list[np.ndarray], scale: tuple=None) -> None:
    """
    Displays a list of images in separate windows.
    Args:
        imageList (list[np.ndarray]): A list of images represented as NumPy arrays to be displayed.
    Returns:
        None
    Each image in the list is displayed in a separate window with a unique title.
    The function waits for a key press before closing each window and proceeding to the next image.
    """
    
    count = 1
    for image in imageList:
        displayString = f"Image {count}"
        print(f"Displaying {displayString}.")
        
        if image is not None:
            # Scaling the output image
            if scale is not None:
                image = cv.resize(image, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_LINEAR)
            
            # Displaying the image, close window on key press
            cv.imshow(displayString, image)
            cv.waitKey()
            cv.destroyAllWindows()
        
        else:
            continue

        count += 1


class Mosaic:
    
    def __init__(self, config):
        
        
        self.scaleDownFactor = 0.5
        
        # Getting the image path
        imgPath = config["Paths"]["imageDir"]
        self.dirPath = os.path.join(os.getcwd(), imgPath)
        
        # Loading SIFT params
        self.siftParams = {
            "nFeatures": config["SIFT"]["nFeatures"],
            "nOctaveLayers": config["SIFT"]["nOctaveLayers"],
            "contrastThreshold": config["SIFT"]["contrastThreshold"],
            "edgeThreshold": config["SIFT"]["edgeThreshold"],
            "sigma": config["SIFT"]["sigma"]
        }

        self.lowes_const = config["FeatureMatching"]["lowes_const"]
        self.RANSAC_THRESH = config["FeatureMatching"]["RANSAC_THRESH"]
        
        # Setting the minimum number of matching key-points
        self.MIN_MATCH_CNT = config["FeatureMatching"]["MIN_MATCH_COUNT"]
        
        # Initializing the Image list
        self.imageList = []
        self.imageCount = 0

        # Initializing the key-point list and the descriptor list
        self.kpList = []
        self.desList = []

    

    def extractFeatures(self, siftParams=None) -> tuple[list, list]:
        """
        Extracts SIFT keypoints and descriptors from each image in the image list.
        Returns:
            tuple[list, list]: A tuple containing two lists:
                - kpList: A list of keypoints for each image.
                - desList: A list of descriptors for each image.
        """

        # Creating the SIFT object
        sift = cv.SIFT.create(
            nfeatures = self.siftParams["nFeatures"],
            nOctaveLayers = self.siftParams["nOctaveLayers"],
            contrastThreshold = self.siftParams["contrastThreshold"],
            edgeThreshold = self.siftParams["edgeThreshold"]
        )

        kpList = []
        desList = []

        # Detecting and computing features and descriptors
        for image in self.imageList:
            
            kp, des = sift.detectAndCompute(image, None)

            # Appending them to a list
            kpList.append(kp)
            desList.append(des)

        return (kpList, desList)
    
    def findMatches(self) -> None:
        """
        Finds matching key-points between consecutive images in the image list.
        Updates matchesList of the object. 
        Returns:
            None
        """

        # BF matcher with default params
        bf = cv.BFMatcher()
        self.matchesList = []

        for idx in range(self.imageCount - 1):
            matches = bf.knnMatch(self.desList[idx], self.desList[idx+1], k=2)

            # Applying Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < self.lowes_const*n.distance:
                    good.append([m])
            
            # Appending good matches to match list
            self.matchesList.append(good)
        
        return None

    def computeHomography(self) -> tuple[list, list]:
        """
        Computes the homographic transform between consecutive images in the image list. The
        function also returns the mask to remove all the outliers.
        Returns:
            tuple[list, list]: A tuple containing two lists:
                - homographicTransformList: List of homographic transforms between consecutive images
                - matchesMaskList: List of masks that can be used to mask the outlying matches
        """

        matchesMaskList = []
        homographicTransformList = []
        for idx in range(self.imageCount - 1):
            
            # Getting the key-points
            kp1 = self.kpList[idx]
            kp2 = self.kpList[idx+1]

            # Error handling in case of insufficient key-points
            if len(self.matchesList[idx]) < self.MIN_MATCH_CNT:
                print("[ERROR]: Not enough matches found between images {} and {}! - {}/{}".format(idx, idx+1, len(self.matchesList[idx]), self.MIN_MATCH_CNT))
                print("Exiting...")
                exit()

            matches = self.matchesList[idx]

            # Extracting source pts and destination pts
            srcPts = np.float32([kp2[m.trainIdx].pt for [m] in matches]).reshape(-1,1,2)
            dstPts = np.float32([kp1[m.queryIdx].pt for [m] in matches]).reshape(-1,1,2)

            # Computing the Homography
            H, mask = cv.findHomography(srcPts, dstPts, cv.RANSAC, self.RANSAC_THRESH)
            
            # Storing the Homography Transform and the Matches mask
            homographicTransformList.append(H)
            matchesMaskList.append(mask.ravel().tolist())
        
        return (homographicTransformList, matchesMaskList)
    
    
    def stitchImages(self, imgList, homographicTransformList) -> np.array:

        # Computing the Transform Matrix w.r.t the first image
        tfNew = np.eye(3) # Identity matrix for the first image
        transformList = []
        for tf in homographicTransformList:
            tfNew = tf @ tfNew
            transformList.append(tfNew)
        
        # Stitching the images
        finalImg = imgList[0]
        translationMatrix = np.eye(3)
        for idx in range(self.imageCount - 1):
            imageIdx = idx + 1 # Getting the image index
            H = transformList[idx] # Getting the Homographic Transform
            targetImg = imgList[imageIdx] # Getting the image to be warped
            numRows, numCols = np.shape(finalImg) # Getting the in-progress mosaic image size

            currMax_X = numCols
            currMin_X = 0

            currMax_Y = numRows
            currMin_Y = 0

            height, width = np.shape(targetImg)

            # Getting all the corner points of the target image
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

            # Computing the translational vectors and consequently, the matrix
            tx = -currMin_X
            ty = -currMin_Y
            translationMatrix[0, 2] = tx
            translationMatrix[1, 2] = ty

            # Translating the in-progress Mosaic image
            finalImg = cv.warpPerspective(finalImg, translationMatrix, (newWidth, newHeight))

            # Accounting for the translation for the target image
            H = translationMatrix @ H

            # Warping the target image
            warpedImg = cv.warpPerspective(targetImg, H, (newWidth, newHeight))

            # Masking the overlapping parts
            _, mask = cv.threshold(warpedImg, 1, 255, cv.THRESH_BINARY_INV)
            warpedFinalImg = cv.bitwise_and(finalImg, mask)

            # Stitching them together
            finalImg = warpedFinalImg + warpedImg
        
        return finalImg

        pass
            

    def InitConfig(self) -> None:
        """
        Function that loads the config file
        """
        # Opening and laoding the config file
        with open("./config/config.yaml") as f:
            config = yaml.safe_load()

        