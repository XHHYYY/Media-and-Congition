import cv2
import numpy as np
from StereoCalibration.codes.stereoconfig import stereoCamera
import scipy.io as scio
import colorDetection as CD




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
 
 
# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
 
    return undistortion_image
 
 
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
 
 
# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    return output
 
 
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
    # ! changed
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    # fb = config.cam_matrix_left[0, 0] * 10000
    doffs = config.doffs
    depthMap = np.divide(fb, np.abs(disparityMap) + doffs)
    reset_index_0 = np.where(depthMap < 0.0)
    depthMap[reset_index_0] = 0
    reset_index_max = np.where(depthMap > 5000.0)
    depthMap[reset_index_max] = 0.0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)



if __name__ == '__main__':
    index = 1
    cam0 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    # cam0.set(3, 1920)
    # cam0.set(4, 1080)
    cam2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cam2.set(3, 1920)
    # cam2.set(4, 1080)
    config = stereoCamera()

    frame0 = []
    frame2 = []
    pre_frame0 = []
    pre_frame2 = []

    while (True):
        pre_frame0 = frame0
        pre_frame2 = frame2
        _, frame2 = cam2.read()
        _, frame0 = cam0.read()
        # cv2.imshow('Capture2', frame2)
        # cv2.imshow('Capture0', frame0)
        iml = frame0
        imr = frame2
        height, width = iml.shape[0:2]
    
        # 读取相机内参和外参
        
    
        # 立体校正
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
        # print(Q) # 视差到深度的映射矩阵
    
        # 绘制等间距平行线，检查立体校正的效果
        line = draw_line(iml_rectified, imr_rectified)
        # cv2.imwrite('./StereoCalibration/images/check.png', line)
    
        # 立体匹配
        # TODO remember to 矫正
        iml_, imr_ = preprocess(iml_rectified, imr_rectified)  # 预处理，一般可以削弱光照不均的影响，不做也可以
        # iml_, imr_ = preprocess(iml, imr)
        original_disp_l, original_disp_r = stereoMatchSGBM(iml_, imr_, True)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
        # kernelSize = 11
        # disp_l = cv2.GaussianBlur(original_disp_l, (kernelSize, kernelSize), 0)
        # disp_r = cv2.GaussianBlur(original_disp_l, (kernelSize, kernelSize), 0)
        disp_l = original_disp_l
        disp_r = original_disp_r
        
        # new_disp_l = cv2.filterSpeckles(disp_l, 100, 10, 50)
        # cv2.imwrite('./StereoCalibration/images/vis_error_l.png', disp_l * 4)
        # cv2.imwrite('./StereoCalibration/images/vis_error_r.png', disp_r)
        # cv2.namedWindow('line', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('line', 1280, 480)
        # cv2.imshow('line', line)
        # cv2.namedWindow('L', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('L', 640, 480)
        # cv2.imshow('L', disp_l * 4)
        # cv2.namedWindow('R', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('R', 640, 480)
        # cv2.imshow('R', disp_r + 255)
        # cv2.waitKey()
    

        # 计算深度图
        depthMap = getDepthMapWithQ(disp_l, Q)
        # depthMap = getDepthMapWithConfig(disp_l, config)
        # depthMap1 = getDepthMapWithQ(disp_l, Q)
        # depthMap2 = getDepthMapWithConfig(disp_l, config)
        # depthMap = (depthMap1 + depthMap2) / 2
        minDepth = np.min(depthMap)
        maxDepth = np.max(depthMap)
        # print(minDepth, maxDepth)
        depthMapVis = (255.0 *(depthMap - minDepth)) / (maxDepth - minDepth)
        depthMapVis = depthMapVis.astype(np.uint8)
        cv2.imshow("DepthMap", depthMapVis)
        
        
        # 颜色检查
        red_detector = CD.colorDetection('red')
        green_detector = CD.colorDetection('green')
        blue_detector = CD.colorDetection('blue')
        yellow_detector = CD.colorDetection('yellow')

        yellow_detector.color_detect(iml_rectified)
        cv2.imshow('yellow_detected', yellow_detector.bgr_img)
        
        k = cv2.waitKey(10)
        # if k == ord('s'):  # 按下s键，进入下面的保存图片操作
        #     # cv2.imwrite('./StereoCalibration/images/LEFT/b8/' + str(index) + ' baseline = ' + str(baselength) + '.jpg', frame0)
        #     # cv2.imwrite('./StereoCalibration/images/RIGHT/b8/' + str(index) + ' baseline = ' + str(baselength) + '.jpg', frame2)
        #     cv2.imwrite('StereoCalibration/images/LEFT/Depth' + str(index) + '.jpg', frame0)
        #     cv2.imwrite('StereoCalibration/images/RIGHT/Depth' + str(index) + '.jpg', frame2)
        #     index += 1
        # elif k == ord('q'):  # 按下q键，程序退出
        #     break
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('./StereoCalibration/images/imr_rectified' + str(index) + '.jpg', iml_rectified)
            # cv2.imwrite('./StereoCalibration/images/depth' + str(index) + '.jpg', depthMapVis)
            scio.savemat('./StereoCalibration/images/Depth_map' + str(index) + '.mat', {'Depth': depthMap})
            index += 1
    cam2.release() # 释放摄像头
    cam0.release() # 释放摄像头
    cv2.destroyAllWindows()# 释放并销毁窗口
