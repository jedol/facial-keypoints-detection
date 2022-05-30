# facial-keypoints-detection
Demo codes for "Unified Convolutional Neural Network for Direct Facial Keypoints Detection"

### Prerequisites
* [caffe](https://github.com/BVLC/caffe)
* [openCV](http://opencv.org/) (read and show image)

### Trained caffemodel
* [SPN160_baseline](https://drive.google.com/file/d/1coXJwbjCMxloxpmw9iiTTJGt1c6BwNSx/view?usp=sharing)
* [SPN160_ResNet](https://drive.google.com/file/d/1fnzqmofhn7mmIjyFvJxM53Bz1FRJqRCQ/view?usp=sharing)

### Usage
```Shell
python tools/demo.py data/1.png data/1.txt models/SPN160_ResNet/SPN160_ResNet_deploy.netpt models/SPN160_ResNet/SPN160_ResNet.caffemodel
```

### Results
<p align="left">
<img src="https://github.com/jedol/facial-keypoints-detection/blob/master/data/1_result.png" alt="demo result 1" width="400px">
</p>
<p align="left">
<img src="https://github.com/jedol/facial-keypoints-detection/blob/master/data/2_result.png" alt="demo result 2" width="400px">
</p>
