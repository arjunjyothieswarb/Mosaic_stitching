import cv2 as cv
import numpy as np
import os

def LoadImages(dirPath: str, scale: tuple=None) -> list[np.ndarray]:
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
    for file in sorted(fileNames):
        filePath = os.path.join(dirPath, file)
        image = cv.imread(filePath)

        if image is not None:
            if scale is not None:
                image = cv.resize(image, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_LINEAR)
            imageList.append(image)
        else:
            print(f"Unable to open {file}! Ignoring {file}")
            continue

    return imageList

def DisplayImages(imageList: list[np.ndarray]) -> None:
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

        # Displaying the image, close window on key press
        cv.imshow(displayString, image)
        cv.waitKey()
        cv.destroyAllWindows()

        count += 1


class Mosaic:
    
    def __init__(self, dirPath):
        
        self.dirPath = dirPath
        self.scaleDownFactor = 0.3
        
        # Loading the images
        self.imageList = LoadImages(self.dirPath, (self.scaleDownFactor, self.scaleDownFactor))

        # Extract the key-points
        self.kpList, self.desList = self.extractFeatures()
    

    def extractFeatures(self) -> tuple[list, list]:
        """
        Extracts SIFT keypoints and descriptors from each image in the image list.
        Returns:
            tuple[list, list]: A tuple containing two lists:
                - kpList: A list of keypoints for each image.
                - desList: A list of descriptors for each image.
        """

        # Creating the SIFT object
        sift = cv.SIFT.create()

        kpList = []
        desList = []

        # Detecting and computing features and descriptors
        for image in self.imageList:
            
            kp, des = sift.detectAndCompute(image, None)

            # Appending them to a list
            kpList.append(kp)
            desList.append(des)

        return (kpList, desList)
    

        

