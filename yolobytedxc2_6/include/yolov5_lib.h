
#pragma once 
#include "dataType.h"
#ifdef __cplusplus
extern "C" 
{
#endif 
 
void * yolov5_trt_create(const char * engine_name);
 
int yolov5_trt_detect(void *h, cv::Mat &img, const float& threshold,std::vector<DetectRes>& det,const float& nms_thresh);
 
void yolov5_trt_destroy(void *h);
 
#ifdef __cplusplus
}
#endif 


