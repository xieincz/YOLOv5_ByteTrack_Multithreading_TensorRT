#ifndef _YOLO_BYTE_API_H_
#define _YOLO_BYTE_API_H_
#include <assert.h>
#include <dirent.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
// #include <pthread.h>//for cpu affinity, Linux only

#include "BYTETracker.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yolov5_lib.h"
using namespace std;

#define CHECK(status)                                \
    do {                                             \
        auto ret = (status);                         \
        if (ret != 0) {                              \
            cerr << "Cuda failure: " << ret << endl; \
            abort();                                 \
        }                                            \
    } while (0)

using namespace nvinfer1;

class YoloByteAPI {
public:
    YoloByteAPI(char* yolo_engine_path, const float& conf_thresh = 0.45, const int& frame_rate = 30, const int& track_buffer = 30, const float& track_thresh = 0.5, const float& high_thresh = 0.6, const float& match_thresh = 0.8, const float& nms_thresh = 0.5);
    ~YoloByteAPI();

    int processVideo(const char *video_path, const char *output_dir, const char *output_file_name, int skip_num);

private:
    void drawResult(const cv::Mat& org_img, const vector<STrack>& output_stracks, const string& output_image_path) const;
    BYTETracker* tracker = nullptr;
    int read_files_in_dir(const char* p_dir_name, vector<string>& file_names);
    void* trt_engine = NULL;
    float conf_thresh;
    float track_thresh;
    float high_thresh;
    float match_thresh;
    float nms_thresh;
    int frame_rate;
    int track_buffer;
    string output_dir;
    VideoWriter videoWriter;
    vector<DetectRes> dets;
    // for multi-thread
    deque<cv::Mat> frames_queue, frames_queue_res;
    deque<vector<DetectRes>> dets_queue;
    deque<vector<STrack>> stracks_queue;
    mutex mtx_frames, mtx_dets, mtx_tracks, mtx_res,mtx_frames_res;
    condition_variable cv_frames, cv_detect, cv_tracks, cv_res;
    thread t_detect, t_track, t_res;
    bool end = false;
    bool read_end = false, detect_end = false, track_end = false, res_end = false;
    void detect();
    void track();
    void res();
};

#endif  // _YOLO_BYTE_API_H_