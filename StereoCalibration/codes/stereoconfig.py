import numpy as np
 
 
# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        # self.cam_matrix_left = np.array([[419.891874377438, -0.846561141862131, 323.944958449294],
        #                                  [0., 420.688855095116,236.244600893985],
        #                                  [0., 0., 1.]])
        self.cam_matrix_left  = np.loadtxt('./StereoCalibration/Params/L_matrix.csv', dtype=float, delimiter=',')
        self.cam_matrix_right = np.loadtxt('./StereoCalibration/Params/R_matrix.csv', dtype=float, delimiter=',').reshape((3, 3))
        # 右相机内参
        # self.cam_matrix_right = np.array([[419.891874377438, -0.846561141862113, 323.944958449294],
        #                                   [0., 420.688855095116, 236.244600893986],
        #                                   [0., 0., 1.]])
 
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        # self.distortion_l = np.array([[-0.0157800580432915, 0.121231420653735, -0.00080265874255202,
        #                                -0.000493406658236845, -0.140415130425101]])
        self.distortion_l = np.loadtxt('./StereoCalibration/Params/L_dist.csv', dtype=float, delimiter=',')
        self.distortion_r = np.loadtxt('./StereoCalibration/Params/R_dist.csv', dtype=float, delimiter=',')
        # self.distortion_r = np.array([[-0.01578005804329, 0.12123142065373, -0.000802658742551806,
        #                                -0.000493406658236739, -0.140415130425096]])
 
        # 旋转矩阵（我的是固定好的，所以转矩阵是单位矩阵）
        # self.R = np.array([[1, 0, 0],
        #                    [0, 1, 0],
        #                    [0, 0, 1]])
        
        self.R = np.loadtxt('./StereoCalibration/Params/Rotation.csv', dtype=float, delimiter=',')
 
        # 平移矩阵
        # self.T = np.array([[-1.39089984389539E-13], [-1.19894454029684E-13], [1.08861855148484E-13]])
        self.T = np.loadtxt('./StereoCalibration/Params/Trans.csv', dtype=float, delimiter=',')
 
        # 焦距
        self.focal_length = 6  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]
 
        # 基线距离
        self.baseline = 60  # 单位：mm， 为平移向量的第一个参数（取绝对值）
        
        self.doffs = 1
        
        self.isRectified = False