# Media and Congnition Project Camera Part

## `CameraIsDone`
Run `CameraIsDone/Top.py` to use stereo camera to get depth and color-detection results online. To Calibrate the camera, see folder `StereoCalibration`.

## `StereoCalibration`
- folder `images` contains stereo images used for stereo-calibration in folder `LEFT` and `Right`.
- Run `Calinration.py` to calibrate the stereo camera.
- Parameters generated will be placed in folder `CameraIsDone/Params`

## Last but not least
There might be some bugs in path, and functions may require parameter `self` if your python version is too high. Solve them by yourself!