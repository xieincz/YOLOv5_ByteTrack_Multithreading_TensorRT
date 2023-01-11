#ifndef _YOLO_BYTE_API_H_
#define _YOLO_BYTE_API_H_
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <pthread.h>
//#include <atomic>

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
    YoloByteAPI(char* yolo_engine_path, const float& conf_thresh = 0.45, const int& frame_rate = 30, const int& track_buffer = 30, const float& track_thresh = 0.5, const float& high_thresh = 0.6, const float& match_thresh = 0.8,const float& nms_thresh=0.5);
    ~YoloByteAPI();
    //vector<vector<float>> detectOneImage(char* image_path);
    int detectImageDirForModel(char* images_dir, char* output_txt_path);
    int detectImageDirForModel(char* images_dir, char* output_txt_path, char* output_image_dir);
    int detectImageDirForModelRenew(char* images_dir, char* output_txt_path);
    int detectImageDirForModelRenew(char* images_dir, char* output_txt_path, char* output_image_dir);

    int detectImageDirForAlgorithm(const char* video_path, const char* polygon_str, const int& frame_count_thresh,int skip_num);  // frame_count_thresh: args['fps'] * args['time_thresh']
int detectImageDirForAlgorithmFast(const char* video_path, const char* polygon_str, const int& frame_count_thresh,int skip_num);
private:
    void drawResultForModel(const cv::Mat& org_img, const vector<STrack>& output_stracks, const string& output_image_path) const;
    void writeResultForModel(FILE* fp, const vector<STrack>& output_stracks, const int& frame_id) const;
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
    vector<DetectRes> dets;
    bool check_crash(const int& px, const int& py, const vector<pair<int, int>>& polygon);
    bool check_crash(const int& px, const int& py, const vector<int>& polygon);
    //for multi-thread
    deque<cv::Mat> frames_queue;
    deque<vector<DetectRes>> dets_queue;
    deque<vector<STrack>> stracks_queue;
    mutex mtx_frames, mtx_dets, mtx_tracks, mtx_res;
    condition_variable cv_frames, cv_detect, cv_tracks, cv_res;
    thread t_detect, t_track, t_res;
    bool end = false;
    //atomic<bool> end;
    bool read_end=false,detect_end = false, track_end = false, res_end = false;
    void detect();
    void track();
    void res();
    set<int> algorithm_data;
    map<int, int> record_id;
    int frame_count_thresh;
    vector<int> polygon;
};

#endif  // _YOLO_BYTE_API_H_