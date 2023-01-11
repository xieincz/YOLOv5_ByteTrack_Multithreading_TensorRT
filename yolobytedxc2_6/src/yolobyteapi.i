%module yolobyteapi
%{
#include "yolobyteapi.h"
%}

using namespace std;

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