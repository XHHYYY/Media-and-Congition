import numpy as np
import cv2
import scipy.io as scio

import depth_map_utils
'''
Fast fill with Gaussian blur @90Hz (paper result)
fill_type = 'fast'
extrapolate = True
blur_type = 'gaussian'

Fast Fill with bilateral blur, no extrapolation @87Hz (recommended)
fill_type = 'fast'
extrapolate = False
blur_type = 'bilateral'

Multi-scale dilations with extra noise removal, no extrapolation @ 30Hz
fill_type = 'multiscale'
extrapolate = False
blur_type = 'bilateral'
'''
def depth_show(winname, DepthMap):
    minDepth = np.min(DepthMap)
    maxDepth = np.max(DepthMap)
    depthMapVis = (255.0 *(DepthMap - minDepth)) / (maxDepth - minDepth)
    depthMapVis = depthMapVis.astype(np.uint8)
    cv2.imshow(winname, depthMapVis)


class PostProcess():
    #Fast Fill with bilateral blur, no extrapolation
    def __init__(self):
        self.fill_type = 'fast'
        self.extrapolate = False
        self.blur_type = 'bilateral'

    def process(self, depth_img):
        # full_in_fast and full_in_multiscale take grayscale image as input
        minDepth = np.min(depth_img)
        maxDepth = np.max(depth_img)
        depthMapVis = (255.0 *(depth_img - minDepth)) / (maxDepth - minDepth)
        depthMapVis = depthMapVis.astype(np.uint8)

        projected_depths = np.float32(depthMapVis / 256.0)
        if self.fill_type == 'fast':
            final_depths = depth_map_utils.fill_in_fast(
                projected_depths, max_depth=maxDepth, extrapolate=self.extrapolate, blur_type=self.blur_type)
        elif self.fill_type == 'multiscale':
            final_depths, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depths, max_depth=maxDepth, extrapolate=self.extrapolate, blur_type=self.blur_type,
                show_process=False)
        else:
            raise ValueError('Invalid fill_type {}'.format(self.fill_type))
        return final_depths

if __name__ == '__main__':
    depth_img_mat = scio.loadmat("./StereoCalibration/images/Depth_map1.mat")
    depth_img = depth_img_mat['Depth']
    Processer = PostProcess()
    depth_show('Depth', depth_img)
    processed_img = Processer.process(depth_img=depth_img)
    depth_show('processed', processed_img)
    cv2.waitKey()
    cv2.destroyAllWindows()