import numpy as np

# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左右相机矩阵
        self.cam_matrix_left  = np.loadtxt('./CameraIsDone/Params/L_matrix.csv', dtype=float, delimiter=',')
        self.cam_matrix_right = np.loadtxt('./CameraIsDone/Params/R_matrix.csv', dtype=float, delimiter=',').reshape((3, 3))
        
        # 左右畸变矩阵
        self.distortion_l = np.loadtxt('./CameraIsDone/Params/L_dist.csv', dtype=float, delimiter=',')
        self.distortion_r = np.loadtxt('./CameraIsDone/Params/R_dist.csv', dtype=float, delimiter=',')
        
        # 旋转矩阵
        self.R = np.loadtxt('./CameraIsDone/Params/Rotation.csv', dtype=float, delimiter=',')
 
        # 平移矩阵
        self.T = np.loadtxt('./CameraIsDone/Params/Trans.csv', dtype=float, delimiter=',')

        # 视差偏置(?)
        self.doffs = 1
