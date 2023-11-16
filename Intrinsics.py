
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = "/huser/users/trainee/mayank/data/exp4zhang/right"

def get_Intrinsics(Path, Rows, Columns):
    '''
    Returns the Intrinsic Matrix of the camera for which the images with chessboard pattern are in the given path.
    Arguments -- Path -> Path to folder containing all the images with chessboard patterns.
                 Rows -> Number of rows in chessboard images.
                 Columns -> Number of columns in chessboard images.

    Returns -- ret -> Root Mean squared Error.
               K -> Intrinsic Matrix for the given images. 
               dist -> distortion parameters for all the images.
               rvec -> In total n rotation matrices for all the chessboard planes.
               tvec -> In total n translation vectors for all chessboard plane origins. 
               Also plots all the images with identified corners along with image name before it. 
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj = np.zeros((Columns * Rows,3),np.float32)
    obj[:,:2] = np.mgrid[0:Rows, 0:Columns].T.reshape(-1,2)
    obj_points = []
    img_points = []
    for img_name in sorted(os.listdir(PATH)):
        img_name = os.path.join(PATH, img_name)
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print(img_name)
        # Find the chess board corners
        ret, crnrs = cv2.findChessboardCorners(gray, (Rows, Columns))
        
        if ret:
            obj_points.append(obj)
            
            window_size = (11, 11)
            zero_zone = (-1, -1)
            crnrs = cv2.cornerSubPix(gray, crnrs, window_size, zero_zone, criteria)
            img_points.append(crnrs)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (Rows,Columns), crnrs, True)
            plt.figure(figsize=(30,30))
            plt.imshow(img)
            plt.show()
    
    ret, K, dist, rvec, tvec = cv2.calibrateCamera(41*obj, img, gray.shape[::-1],None,None)
    return ret, K, dist, rvec, tvec

