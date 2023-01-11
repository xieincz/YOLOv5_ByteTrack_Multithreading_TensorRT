#include "yolobyteapi.h"
static Logger gLogger;

int YoloByteAPI::read_files_in_dir(const char *p_dir_name, vector<string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }
    struct dirent *p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    std::sort(file_names.begin(), file_names.end());
    return 0;
}

YoloByteAPI::YoloByteAPI(char *yolo_engine_path, const float &conf_thresh, const int &frame_rate, const int &track_buffer, const float &track_thresh, const float &high_thresh, const float &match_thresh, const float &nms_thresh) {
    trt_engine = yolov5_trt_create(yolo_engine_path);
    this->track_thresh = track_thresh;
    this->high_thresh = high_thresh;
    this->match_thresh = match_thresh;
    this->frame_rate = frame_rate;
    this->track_buffer = track_buffer;

    tracker = new BYTETracker(frame_rate, track_buffer, track_thresh, high_thresh, match_thresh);
    this->conf_thresh = conf_thresh;
    this->nms_thresh = nms_thresh;
    // for multi-thread
    t_detect = thread(&YoloByteAPI::detect, this);
    t_track = thread(&YoloByteAPI::track, this);
    t_res = thread(&YoloByteAPI::res, this);
}

YoloByteAPI::~YoloByteAPI() {
    yolov5_trt_destroy(trt_engine);
    delete tracker;
    end = true;
    cv_frames.notify_all();
    cv_detect.notify_all();
    cv_tracks.notify_all();
    cv_res.notify_all();
    if (t_detect.joinable())
        t_detect.join();
    if (t_track.joinable())
        t_track.join();
    if (t_res.joinable())
        t_res.join();
}

void YoloByteAPI::drawResultForModel(const cv::Mat &org_img, const vector<STrack> &output_stracks, const string &output_image_path) const {
    cv::Mat src_img = org_img.clone();
    for (const STrack &d : output_stracks) {
        const vector<float> &tlwh = d.tlwh;
        cv::rectangle(src_img, cv::Rect2f(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), cv::Scalar(0, 255, 0), 2);
        cv::putText(src_img, to_string(d.track_id), cv::Point2f(tlwh[0], tlwh[1]), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(output_image_path, src_img);
    cout << "write image to " << output_image_path << endl;
}

bool YoloByteAPI::check_crash(const int &px, const int &py, const vector<pair<int, int>> &polygon) {
    bool is_in = false;
    int l = polygon.size(), next_i = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0, x = 0;
    for (int i = 0; i < l; i++) {
        next_i = (i + 1) % l;
        x1 = polygon[i].first;
        y1 = polygon[i].second;
        x2 = polygon[next_i].first;
        y2 = polygon[next_i].second;
        if (((x1 == px) && (y1 == py)) || ((x2 == px) && (y2 == py))) {  // if point is on vertex
            is_in = true;
            break;
        }
        if (min(y1, y2) < py && py < max(y1, y2)) {  // find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1);
            if (x == px) {  // if point is on edge
                is_in = true;
                break;
            } else if (x > px) {  // if point is on right side of edge
                is_in = !is_in;
            }
        }
    }
    return is_in;
}

bool YoloByteAPI::check_crash(const int &px, const int &py, const vector<int> &polygon) {
    bool is_in = false;
    int l = polygon.size(), next_i = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0, x = 0;
    for (int i = 0; i < l; i += 2) {
        next_i = (i + 2) % l;
        x1 = polygon[i];
        y1 = polygon[(i + 1) % l];
        x2 = polygon[next_i];
        y2 = polygon[(next_i + 1) % l];
        if (((x1 == px) && (y1 == py)) || ((x2 == px) && (y2 == py))) {  // if point is on vertex
            is_in = true;
            break;
        }
        if (min(y1, y2) < py && py < max(y1, y2)) {  // find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1);
            if (x == px) {  // if point is on edge
                is_in = true;
                break;
            } else if (x > px) {  // if point is on right side of edge
                is_in = !is_in;
            }
        }
    }
    return is_in;
}

int YoloByteAPI::detectImageDirForAlgorithmFast(const char *video_path, const char *polygon_str, const int &frame_count_thresh, int skip_num) {
    read_end = false, detect_end = false, track_end = false, res_end = false;
    this->frame_count_thresh = const_cast<int &>(frame_count_thresh);

    cv::VideoCapture video_cap(video_path);
    int i = 0, width = video_cap.get(cv::CAP_PROP_FRAME_WIDTH), height = video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    record_id.clear();
    algorithm_data.clear();
    polygon.clear();
    string reg = "\\d+(\\.\\d+)?", polygon_string = polygon_str;
    regex Pattern(reg);
    smatch result;
    string::const_iterator iterStart = polygon_string.begin();
    string::const_iterator iterEnd = polygon_string.end();
    while (regex_search(iterStart, iterEnd, result, Pattern)) {
        if (i & 1) {                                                      // 奇数
            polygon.emplace_back((int)(stof(result[0].str()) * height));  // y
        } else {
            polygon.emplace_back((int)(stof(result[0].str()) * width));  // x
        }
        ++i;
        // polygon.emplace_back(result[0]*width);
        iterStart = result[0].second;  // 更新搜索起�?�位�??,搜索剩下的字符串
    }

    cv::Mat src_img;
    while (video_cap.read(src_img)) {
        for (int i = 1; i <= skip_num; ++i) {
            if (!video_cap.read(src_img))
                break;
        }
        // unique_lock<mutex> lck(mtx_frames);
        mtx_frames.lock();
        frames_queue.push_back(src_img);
        mtx_frames.unlock();
        // lck.unlock();
        cv_frames.notify_one();
    }
    read_end = true;
    cv_frames.notify_all();

    unique_lock<mutex> lck(mtx_res);
    // cout<<"cv_res.wait(lck, [&] { return res_end; });"<<endl;
    cv_res.wait(lck, [&] { return res_end; });
    // cout<<"out wake up ed"<<endl;
    int res = algorithm_data.size();
    res_end = false;
    lck.unlock();
    return res;
}

void YoloByteAPI::detect() {
    /*
    //for cpu affinity, Linux only
    //Create a cpu_set_t object representing a set of CPUs. Clear it and mark
    //only CPU i as set.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);//this thread will be bind to cpu 0
    int rc = pthread_setaffinity_np(t_detect.native_handle(),sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }*/
    cv::Mat src_img;
    while (!end) {
        unique_lock<mutex> lck(mtx_frames);
        if (!read_end)
            cv_frames.wait(lck, [this] { return !frames_queue.empty() || read_end || end; });
        if (end) {
            lck.unlock();
            return;
        }
        if (read_end && frames_queue.empty()) {
            lck.unlock();
            read_end = false;
            detect_end = true;
            cv_detect.notify_all();
            continue;
        }
        src_img = frames_queue.front().clone();
        frames_queue.pop_front();
        mtx_dets.lock();
        yolov5_trt_detect(trt_engine, src_img, conf_thresh, dets, nms_thresh);
        lck.unlock();
        /*FILE *fp = fopen("/project/ev_sdk/src/testdet.txt", "a");
        for(DetectRes& d:dets){
            fprintf(fp, "%d,%d,%d,%d,%d,%f\n",d.track_id, d.tx, d.ty, d.w, d.h, d.confidence);
        }
        fclose(fp);*/
        // unique_lock<mutex> lck2(mtx_dets);

        dets_queue.emplace_back(dets);
        mtx_dets.unlock();
        cv_detect.notify_one();
    }
}

void YoloByteAPI::track() {
    /*
    //for cpu affinity, Linux only
    //Create a cpu_set_t object representing a set of CPUs. Clear it and mark
    //only CPU i as set.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);//this thread will be bind to cpu 2
    int rc = pthread_setaffinity_np(t_track.native_handle(),sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }*/
    vector<STrack> output_stracks;
    while (!end) {
        unique_lock<mutex> lck(mtx_dets);
        if (!detect_end)
            cv_detect.wait(lck, [this] { return !dets_queue.empty() || detect_end || end; });
        if (end) {
            lck.unlock();
            return;
        }
        if (detect_end && dets_queue.empty()) {
            lck.unlock();
            detect_end = false;
            track_end = true;
            cv_tracks.notify_all();
            continue;
        }
        output_stracks = tracker->update(dets_queue.front());
        dets_queue.pop_front();

        /*FILE *fp = fopen("/project/ev_sdk/src/testtrack.txt", "a");
        for(STrack& d:output_stracks){
            const vector<float> &tlwh = d.tlwh;
            fprintf(fp, "%d,%d,%d,%d,%d,%f\n",int(d.track_id), max(int(tlwh[0]), 0), max(int(tlwh[1]), 0), int(tlwh[2]), int(tlwh[3]), d.score);
        }
        fclose(fp);*/
        // unique_lock<mutex> lck2(mtx_tracks);
        mtx_tracks.lock();
        stracks_queue.emplace_back(output_stracks);
        lck.unlock();
        mtx_tracks.unlock();
        cv_tracks.notify_one();
    }
}

void YoloByteAPI::res() {
    /*
    //for cpu affinity, Linux only
    //Create a cpu_set_t object representing a set of CPUs. Clear it and mark
    //only CPU i as set.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);//this thread will be bind to cpu 3
    int rc = pthread_setaffinity_np(t_res.native_handle(),sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }*/
    set<int> interrupt_ids;
    while (!end) {
        unique_lock<mutex> lck(mtx_tracks);
        if (!track_end)
            cv_tracks.wait(lck, [this] { return !stracks_queue.empty() || track_end || end; });
        if (end) {
            lck.unlock();
            return;
        }
        if (track_end && stracks_queue.empty()) {
            lck.unlock();
            track_end = false;
            res_end = true;
            record_id.clear();
            cv_res.notify_all();
            delete tracker;
            tracker = new BYTETracker(frame_rate, track_buffer, track_thresh, high_thresh, match_thresh);
            continue;
        }
        vector<STrack> output_stracks = stracks_queue.front();
        stracks_queue.pop_front();

        for (const auto &it : record_id) {
            interrupt_ids.emplace(it.first);  // it.first is the key
        }
        for (const STrack &d : output_stracks) {
            const vector<float> &tlwh = d.tlwh;
            int track_id = d.track_id;
            if (check_crash(tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, polygon)) {
                if (record_id.find(track_id) == record_id.end()) {  // not found
                    record_id[track_id] = 1;
                } else {  // found
                    ++record_id[track_id];
                    interrupt_ids.erase(track_id);
                }
            }
        }
        lck.unlock();
        for (const auto &it : interrupt_ids) {
            record_id.erase(it);
        }
        for (const auto &it : record_id) {
            if (it.second >= frame_count_thresh) {
                algorithm_data.emplace(it.first);  // it.first is the key witch means person_id
            }
        }
        interrupt_ids.clear();
    }
}