import cv2
import numpy as np
from stereoconfig import stereoCamera

class Depth():
    
    def __init__(self) -> None:
        self.DepthMap = []
        self.depthMapVis = []

    # 预处理
    def preprocess(img1, img2):
        # 彩色图->灰度图
        if(img1.ndim == 3):
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
        if(img2.ndim == 3):
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
        # 直方图均衡
        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)
    
        return img1, img2


    # 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
    # @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
    def getRectifyTransform(height, width, config):
        # 读取内参和外参
        left_K = config.cam_matrix_left
        right_K = config.cam_matrix_right
        left_distortion = config.distortion_l
        right_distortion = config.distortion_r
        R = config.R
        T = config.T
    
        # 计算校正变换
        # ! original alpha = 0
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0)
    
        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
    
        return map1x, map1y, map2x, map2y, Q


    # 畸变校正和立体校正
    def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    
        return rectifyed_img1, rectifyed_img2


    # 视差计算
    def stereoMatchSGBM(left_image, right_image, down_scale=False):
        # SGBM匹配参数设置
        if left_image.ndim == 2:
            img_channels = 1
        else:
            img_channels = 3
        blockSize = 3
        paraml = {'minDisparity': 0,
                'numDisparities': 72,
                'blockSize': blockSize,
                'P1': 2 * img_channels * blockSize ** 2,
                'P2': 256 * img_channels * blockSize ** 2,
                'disp12MaxDiff': 10,
                'preFilterCap': 63,
                'uniquenessRatio': 10, # 5~15
                'speckleWindowSize': 100, # 50~200
                'speckleRange': 10, # 1~2
                'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                }
        # paraml = {'minDisparity': 8,
        #         'numDisparities': 96,
        #         'blockSize': blockSize,
        #         'P1': 8 * img_channels * blockSize ** 2,
        #         'P2': 128 * img_channels * blockSize ** 2,
        #         'disp12MaxDiff': 10,
        #         'preFilterCap': 63,
        #         'uniquenessRatio': 10, # 5~15
        #         'speckleWindowSize': 100, # 50~200
        #         'speckleRange': 32, # 1~2
        #         'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
        #         }
    
        # 构建SGBM对象
        left_matcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml
        paramr['minDisparity'] = -paraml['numDisparities']
        right_matcher = cv2.StereoSGBM_create(**paramr)
    
        # 计算视差图
        size = (left_image.shape[1], left_image.shape[0])
        if down_scale == False:
            disparity_left = left_matcher.compute(left_image, right_image)
            disparity_right = right_matcher.compute(right_image, left_image)
    
        else:
            left_image_down = cv2.pyrDown(left_image)
            right_image_down = cv2.pyrDown(right_image)
            factor = left_image.shape[1] / left_image_down.shape[1]
    
            disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
            disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = factor * disparity_left
            disparity_right = factor * disparity_right
    
        # 真实视差（因为SGBM算法得到的视差是×16的）
        trueDisp_left = disparity_left.astype(np.float32) / 16.
        trueDisp_right = disparity_right.astype(np.float32) / 16.
    
        return trueDisp_left, trueDisp_right


    def getDepthMapWithQ(disparityMap : np.ndarray, Q : np.ndarray) -> np.ndarray:
        points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
        depthMap = points_3d[:, :, 2]
        reset_index_0 = np.where(depthMap < 0.0)
        depthMap[reset_index_0] = 0
        reset_index_max = np.where(depthMap > 5000.0)
        depthMap[reset_index_max] = 0.0
        reset_index2 = np.where(disparityMap < 0.0)
        depthMap[reset_index2] = 0
        # reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
        # depthMap[reset_index] = 0
    
        return depthMap.astype(np.float32)


    def getDepthMapWithConfig(disparityMap : np.ndarray, config : stereoCamera) -> np.ndarray:
        fb = config.cam_matrix_left[0, 0] * (-config.T[0])
        doffs = config.doffs
        depthMap = np.divide(fb, np.abs(disparityMap) + doffs)
        reset_index_0 = np.where(depthMap < 0.0)
        depthMap[reset_index_0] = 0
        reset_index_max = np.where(depthMap > 5000.0)
        depthMap[reset_index_max] = 0.0
        reset_index2 = np.where(disparityMap < 0.0)
        depthMap[reset_index2] = 0
        return depthMap.astype(np.float32)


    def get_Depth(self, iml: np.ndarray, imr: np.ndarray, mode: str) -> None:

        height, width = iml.shape[0:2]
        
        # 读取相机内参和外参
        config = stereoCamera()

        # 立体校正
        map1x, map1y, map2x, map2y, Q = self.getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = self.rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    
        # 立体匹配
        iml_, imr_ = self.preprocess(iml_rectified, imr_rectified)  # 预处理，一般可以削弱光照不均的影响，不做也可以
        original_disp_l, original_disp_r = self.stereoMatchSGBM(iml_, imr_, True) 
        disp_l = original_disp_l
        disp_r = original_disp_r
    

        # 计算深度图
        if mode == 'Q':
            self.DepthMap = self.getDepthMapWithQ(disp_l, Q)
        elif mode == 'C':
            self.DepthMap = self.getDepthMapWithConfig(disp_l, config)
        else:
            depthMap1 = self.getDepthMapWithQ(disp_l, Q)
            depthMap2 = self.getDepthMapWithConfig(disp_l, config)
            self.DepthMap = (depthMap1 + depthMap2) / 2

        minDepth = np.min(self.DepthMap)
        maxDepth = np.max(self.DepthMap)
        depthMapVis = (255.0 *(self.DepthMap - minDepth)) / (maxDepth - minDepth)
        self.depthMapVis = depthMapVis.astype(np.uint8)

