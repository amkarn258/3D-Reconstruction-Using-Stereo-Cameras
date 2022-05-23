#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:03:34 2022

@author: dp
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cameraOrientation
import open3d as o3d

KL = np.array([[3.69825407e+03/2, 0.00000000e+00, 2.64396863e+03/2],
       [0.00000000e+00, 3.70548481e+03/2, 1.89670710e+03/2],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
KR = np.array([[3.63379404e+03/2, 0.00000000e+00, 2.67271355e+03/2],
       [0.00000000e+00, 3.63448202e+03/2, 1.92521006e+03/2],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = None
imgL = cv2.imread("C:/Users/karnz/OneDrive/Desktop/SAC_Intern/Codefiles/Stereo_Images/DSC_0124.JPG")
imgR = cv2.imread("C:/Users/karnz/OneDrive/Desktop/SAC_Intern/Codefiles/Stereo_Images/DSC_0163.JPG")
imgR = cv2.resize(imgR, (2464, 1632))
imgL = cv2.resize(imgL, (2464, 1632))
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

def getEssential(img1, img2, IL, IR):
    '''
    Computes the Essential Matrix and the mask of homography.
    Arguments -- img1 -> The left Image.
                 img2 -> The right Image.
                 IL -> Intrinsic Matrix of left camera.
                 IR -> Intrinsic Matrix of right camera.
    Returns -- E -> Essential Matrix
               mask -> Mask containing the homography.
               pts1 -> Points on image1.
               pts2 -> Points on image2.
    '''
    grayL = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image_size = grayL.shape
    
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(grayL, None)
    kp2, desc2 = sift.detectAndCompute(grayR, None)
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    E, mask = cv2.findEssentialMat(pts1, pts2, IL, method=cv2.FM_RANSAC, prob=0.99,
                                   threshold=0.4, mask=None)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return E, mask, pts1, pts2

def getFundamentalMat(E, IL, IR):
    '''
    Computes the fundamental matrix using Essential and Intrinsic Matrices
    Arguments : E -> Essential Matrix
                IL -> Left Intrinsic Matrix
                IR -> Right Intrinsic Matrix
    Returns : F -> Fundamental Matrix
    '''
    K_invL = np.linalg.inv(IL)
    K_invR = np.linalg.inv(IR)
    return K_invL.T @ E @ K_invR


def decompose_essential_matrix(E,K,pts1,pts2):
    '''
    Computes the Rotation and translation from essential matrix. 
    Arguments -- E -> Essential Matrix
                 K -> Intrinsic Matrix
                 pts1 -> Corresponding Points in Image1
                 pts2 -> Corresponding Points in Image2
    Returns -- R -> The Rotation Matrix
               t -> The translation Vector 
    '''
    [U, D, V] = np.linalg.svd(E)
    diag_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    new_E = U @ diag_arr @ V
    [U, D, V] = np.linalg.svd(new_E)
    Y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = - U @ Y @ V
    R2 = - U @ Y.T @ V
    t = U[:, 2].reshape(3, 1)
    R_mat = np.array([R1, R1, R2, R2])
    T_mat = np.array([t, -t, t, -t])
    P1 = np.zeros((3, 4))
    P1[:, :3] = np.eye(3)
    P1 = K @ P1
    print(R1, "\n", R2)
    for i in range(4):
        P2 = np.concatenate((R_mat[i], T_mat[i]), axis=1)
        P2 = K @ P2
        world_pts = cv2.triangulatePoints(P1, P2, pts1, pts2)
        X, Y, Z = world_pts[:3, :] / world_pts[3, :]
        Z_ = R_mat[i][2, 0] * X + R_mat[i][2, 1] * Y + R_mat[i][2, 2] * Z + T_mat[i][2]
        print(len(np.where(Z < 0)[0]), len(np.where(Z_ < 0)[0]))
        if len(np.where(Z < 0)[0]) == 0:
            R = R_mat[i]
            t = T_mat[i]
            break
    return R,t

def drawlines(img1,img2,lines,pts1,pts2):
    '''
    Computes the Images with drawn epipolar lines and corresponding Points.
    Arguments -- img1 -> The image on which it is intended to draw the lines.
                 img2 -> The image on which the corresponding points are there for which the lines are drawn on image1.
                 lines -> The lines obtained after applying epipolar geometry.
                 pts1 -> Corresponding Points in Image1.
                 pts2 -> Corresponding Points in Image2.
    Returns -- img1 -> Image after drawing lines.
               img2 -> Image after drawing corresponding Points. 
    '''
    r,c,ch = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2

def rectification(img1, img2, IL, IR, R, t):
    '''
    Computes the rectified Images.
    Arguments -- img1 -> Left Image
                 img2 -> right image
                 IL -> Intrinsic Matrix of Left Camera.
                 IR -> Intrinsic Matrix of Right Camera.
                 R -> Rotation Matrix of Right Camera w.r.t Left Camera.
                 t -> translation vector between cameras (left to right).
    Returns -- img1_rectified -> Rectified Left Image.
               img2_rectified -> Rectified Right Image. 
    '''
    image_size = img1.shape
    R1,R2,P1,P2= cv2.stereoRectify(IL,None,IR,None,(image_size[1],image_size[0]),R,t,flags = cv2.CALIB_ZERO_DISPARITY)[:4]
    #print(R1 @ R2.T) # these gives the rotation between the two camera
    mapx1,mapy1 = cv2.initUndistortRectifyMap(IL,None,R1,P1,(image_size[1],image_size[0]),cv2.CV_16SC2)
    mapx2,mapy2 = cv2.initUndistortRectifyMap(IL,None,R2,P2,(image_size[1],image_size[0]),cv2.CV_16SC2)
    img1_rectified = cv2.remap(img1,mapx1,mapy1,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    img2_rectified = cv2.remap(img2,mapx2,mapy2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    
    img1_rectified = cv2.pyrDown(img1_rectified)
    img2_rectified = cv2.pyrDown(img2_rectified)
    
    
    return img1_rectified, img2_rectified

'''
Calculating the disparity Map
'''
min_disparity =  -64
max_disparity = 180
num_disparity = max_disparity - min_disparity
SADWindowSize = 5
uniqueness = 3
speckle_windows_size = 5
speckle_range = 5
P1_ = 64
P2_ = 512

left_matcher = cv2.StereoSGBM_create(minDisparity=min_disparity,numDisparities=num_disparity,blockSize=SADWindowSize
                               ,P1= P1_,P2=P2_,uniquenessRatio=uniqueness,disp12MaxDiff=2,
                                speckleWindowSize=speckle_windows_size,speckleRange=speckle_range)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


def getDisparity(left_matcher, right_matcher, img1_rectified, img2_rectified):
    '''
    Computes the Disparity Map for the rectified Images. 
    Arguments -- left_matcher -> Left Disparity Matcher.
                 right_matches -> Right Disparity Matcher.
                 img1_rectified -> Left rectified Image
                 img2_rectified -> Right rectified Image
    Returns -- left_disparity -> Disparity map of left image w.r.t right image.
               right_disparity -> Disparity map of right image w.r.t left image. 
    '''
    left_disparity = left_matcher.compute(img1_rectified, img2_rectified)
    right_disparity = right_matcher.compute(img2_rectified, img1_rectified)
    
    return left_disparity, right_disparity

def get_filtered_disparity(left_disparity, right_disparity, img1_rectified, img2_rectified, sigma = 1.5, 
                           lambda_ = 8000):
    '''
    Computes the Filtered Disparity Map for the rectified Images. 
    Arguments -- left_disparity -> Disparity map of left image w.r.t right image.
                 right_disparity -> Disparity map of right image w.r.t left image.
                 img1_rectified -> Left rectified Image
                 img2_rectified -> Right rectified Image
    Returns -- filtered_disparity -> Filtered disparity map after applying wls filter.
    '''
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(lambda_)
    wls.setSigmaColor(sigma)
    filtered_disparity = wls.filter(left_disparity, img1_rectified,disparity_map_right = right_disparity)
    cv2.filterSpeckles(filtered_disparity,0,400,max_disparity-5)
    _,filtered_disparity = cv2.threshold(filtered_disparity,0,max_disparity*16,cv2.THRESH_TOZERO)
    filtered_disparity = (filtered_disparity/16).astype(np.uint8)
    
    return filtered_disparity

def getReprojected3D(filtered_disparity, IL, IR, b, img1_rectified):
    '''
    Computes the 3d coordinates of all the points common in both images and their respective colors.
    Arguments -- filtered_disparity -> Filtered disparity map after applying wls filter.
                 IL -> Intrinsic Matrix of the left camera.
                 IR -> Intrinsic Matrix of the right camera.
                 b -> baseline between two cameras.
                 img1_rectified -> Rectified Left image.
    Returns -- points -> A (n, 3) shaped array containing the 3d points of n points common in left and right image
               color -> A (n,3) shaped array containing the corresponding colors for all the points. (RGB FORMAT) 
    '''
    Q = np.float32([[1,0,0,-IL[0,2]],
                [0,1,0,-IL[1,2]],
                [0,0,0,IL[0,0]],
                [0,0,-1/b,(IL[0,2]-IR[0,2])/b]])
    
    points = cv2.reprojectImageTo3D(filtered_disparity,Q)
    points = points.reshape(-1,3)
    color = img1_rectified.reshape(-1,3)
    color = np.flip(color,axis = 1)/255
    return points, color

def main():
    IL = np.array([[3.69825407e+03/2, 0.00000000e+00, 2.64396863e+03/2],
       [0.00000000e+00, 3.70548481e+03/2, 1.89670710e+03/2],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    IR = np.array([[3.63379404e+03/2, 0.00000000e+00, 2.67271355e+03/2],
       [0.00000000e+00, 3.63448202e+03/2, 1.92521006e+03/2],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coeff = None
    imgL = cv2.imread("C:/Users/karnz/OneDrive/Desktop/SAC_Intern/Codefiles/Stereo_Images/DSC_0124.JPG")
    imgR = cv2.imread("C:/Users/karnz/OneDrive/Desktop/SAC_Intern/Codefiles/Stereo_Images/DSC_0163.JPG")
    imgR = cv2.resize(imgR, (2464, 1632))
    imgL = cv2.resize(imgL, (2464, 1632))
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    E, mask, pts1, pts2 = getEssential(imgL, img2 = imgR, IL = IL, IR = IR)
    R, R_, t = cv2.decomposeEssentialMat(E)
    F = getFundamentalMat(E, IL, IR)
    print("Rotation Matrix between Cameras is given as : " + str(R))
    print("Translation between Cameras is given as : " + str(t))
    print("Fundamental Matrix is given as : " + str(F))
    print("Essential Matrix is given as : " + str(E))
    imgL_rectified, imgR_rectified = rectification(grayL, grayR, IL, IR, R, t)
    stacked = np.stack((imgL_rectified, imgR_rectified, imgR_rectified), axis = 2)
    plt.figure(figsize=(30,30))
    plt.imshow(stacked)
    left_disparity, right_disparity = getDisparity(left_matcher, right_matcher, imgL_rectified, imgR_rectified)
    filtered_disparity = get_filtered_disparity(left_disparity, right_disparity, imgL_rectified, imgR_rectified)
    plt.figure(figsize=(30,30))
    plt.imshow(filtered_disparity)
    #define baseline
    b = 12
    imgL_clr_rectified, imgR_clr_rectified = rectification(imgL, imgR, IL, IR, R, t)
    points, colors = getReprojected3D(filtered_disparity, IL, IR, b, imgL_clr_rectified)
    scale = 9.63
    tcm = t * scale * 2.54
    #cameraOrientation.cameraOrientation(R, tcm)
    '''Open3D is required for the following code, it'd convert these points and colors into a ply file which
     can be visualized in meshlab or open3D'''
    
    xyzrbg = np.concatenate((points,colors),axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrbg[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrbg[:,3:])
    ptsf = []
    clrs = []
    for i in range(points.shape[0]):
        if (xyzrbg[i,0]>0 and xyzrbg[i,2]!=-24831.056640625):
            ptsf.append(xyzrbg[i])
        #clrs.append(color[i])
    ptsf = np.array(ptsf)
    #ptsf.shape
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptsf[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(ptsf[:,3:])
    #o3d.io.write_point_cloud('C:/Users/mayank/OneDrive/Desktop/SAC_Intern/lunar_kind/data15.ply',pcd)
    o3d.visualization.draw_geometries([pcd])
if __name__=="__main__":
    main()