import cv2 as cv
import numpy as np
import os


def LoadImages(dirPath: str) -> list[np.ndarray]:
    """
    Loads all image files from the specified directory and returns them as a list of numpy arrays.
    Args:
        dirPath (str): The path to the directory containing image files.
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
        filePath = os.join(dirPath, file)
        image = cv.imread(filePath)

        if image is not None:
            imageList.append(image)
        else:
            print(f"Unable to open {file}! Ignoring {file}")
            continue

    return imageList





class Mosaic:
    def __init__(self):
        pass
