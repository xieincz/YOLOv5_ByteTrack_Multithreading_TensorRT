# YOLOv5_ByteTrack_Multithreading_TensorRT
 Multi-thread tracking of YOLOv5 and ByteTrack implemented by C++, accelerated by TensorRT.

检测器采用的是 YOLOv5 （可以轻松替换成其他同类的检测器），追踪器采用的是 ByteTrack 。和其他同类项目不同的是加入了多线程处理以及用 C++和 TensorRT 加快推理速度。一个线程负责读取来自视频文件（可以轻松更改为摄像头）的帧，一个线程负责用 YOLOv5 得到检测框，一个线程用 ByteTrack 给各个检测框 reid ，还有一个线程负责将结果绘制到视频文件中。在多 CPU 核心（>=3 核）的设备上的效果比目前其他同类的项目要更快。而且还可以根据需要给各个线程设置 CPU 亲和性（将某线程绑定到某个 CPU 核心，该功能仅限于 Linux 平台）。

此外还利用 SWIG 包装了接口，方便在 python 中像调用一个库一样使用本项目。

**TODO**

- [ ] YOLOv8

## Usage and demo

Click the button shown below to try this project in colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xieincz/YOLOv5_ByteTrack_Multithreading_TensorRT/blob/main/colab/YOLOv5_ByteTrack_Multithreading_TensorRT.ipynb)

note: You can also upload the notebook in the colab folder of this project to the colab for running.

## Acknowledgement

A large part of the code is borrowed from [yolov5_deepsort_tensorrt](https://github.com/cong/yolov5_deepsort_tensorrt), [TensorRTx](https://github.com/wang-xinyu/tensorrtx), [YOLOv5](https://github.com/ultralytics/yolov5) and [ByteTrack](https://github.com/ifzhang/ByteTrack). Thanks for their wonderful works.
