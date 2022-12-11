import cv2
import os
import glob
import numpy as np

leftpath = './StereoCalibration/images/LEFT/b8'
rightpath = './StereoCalibration/images/RIGHT/b8'
CHECKERBOARD = (9,6)  #棋盘格内角点数
square_size = (25,25)   #棋盘格大小，单位mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
imgpoints_l = []    #存放左图像坐标系下角点位置
imgpoints_r = []    #存放左图像坐标系下角点位置
objpoints = []   #存放世界坐标系下角点位置
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[0,:,0] *= square_size[0]
objp[0,:,1] *= square_size[1]

images_l = glob.glob('./StereoCalibration/images/LEFT/*.jpg')
images_r = glob.glob('./StereoCalibration/images/RIGHT/*.jpg')
for ii in range(len(images_l)):
    img_l = cv2.imread(images_l[ii])
    gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(images_r[ii])
    gray_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)   #检测棋盘格内角点
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)
    if ret_l and ret_r:
        objpoints.append(objp)
        corners2_l = cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),criteria) 
        imgpoints_l.append(corners2_l)
        corners2_r = cv2.cornerSubPix(gray_r,corners_r,(11,11),(-1,-1),criteria)
        imgpoints_r.append(corners2_r)
        #img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
        #cv2.imwrite('./ChessboardCornersimg.jpg', img)
ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1],None,None)  #先分别做单目标定
ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1],None,None)

# flags = 0
# # flags |= cv2.CALIB_FIX_ASPECT_RATIO
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# # flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# # flags |= cv2.CALIB_FIX_INTRINSIC
# # flags |= cv2.CALIB_FIX_K1
# # flags |= cv2.CALIB_FIX_K2
# # flags |= cv2.CALIB_FIX_K3
# # flags |= cv2.CALIB_FIX_K4
# # flags |= cv2.CALIB_FIX_K5
# # flags |= cv2.CALIB_FIX_K6
# stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
#                         cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
#     cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], criteria=stereocalib_criteria, flags=flags)   #再做双目标定

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1])   #再做双目标定


print("\nstereoCalibrate :")
print('retval:')
print(retval)
print("\nCamera matrix left :")
print(cameraMatrix1)
print("\ndistCoeffs left  :")
print(distCoeffs1)
print("\ncameraMatrix right :")
print(cameraMatrix2)
print("\ndistCoeffs right :")
print(distCoeffs2)
print("\nR :")
print(R)
print("\nT :")
print(T)
print("\nE :")
print(E)
print("\nF :")
print(F)

np.savetxt('./CameraIsDone/Params/Rotation.csv', R, fmt='%f', delimiter=',')
np.savetxt('./CameraIsDone/Params/Trans.csv', T, fmt='%f', delimiter=',')
np.savetxt('./CameraIsDone/Params/L_matrix.csv', cameraMatrix1, fmt='%f', delimiter=',')
np.savetxt('./CameraIsDone/Params/L_dist.csv', distCoeffs1, fmt='%f', delimiter=',')
np.savetxt('./CameraIsDone/Params/R_matrix.csv', cameraMatrix2, fmt='%f', delimiter=',')
np.savetxt('./CameraIsDone/Params/R_dist.csv', distCoeffs2, fmt='%f', delimiter=',')