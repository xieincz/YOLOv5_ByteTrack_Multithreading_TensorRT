import json
from yolobytedxc2_6 import yolobyteapi
import logging


def init():
    # for yolo
    yolo_engine_path = '/project/ev_sdk/src/cppmodels/yolov5l6.engine
    
    conf_thresh = 0.15
    nms_thresh = 0.52

    # for bytetrack
    frame_rate = 30
    track_buffer = 30
    track_thresh = 0.5
    high_thresh = 0.65
    match_thresh = 0.85
    model = yolobyteapi.YoloByteAPI(yolo_engine_path, conf_thresh, frame_rate,track_buffer, track_thresh, high_thresh, match_thresh, nms_thresh)

    return model


def process_video(handle=None, input_video=None, args=None, **kwargs):

    args = eval(args)

    # 获取徘徊阈值，转换成帧数
    #每获取1帧要跳过几帧
    skip_num=5
    #提前多少检测帧认为一个人是徘徊人员(用于修正跳帧导致的计算误差)
    advance=0

    frame_count_thresh = (args['fps'] * args['time_thresh'])/(1+skip_num)-advance

    model = handle
    cnt = model.detectImageDirForAlgorithmFast(input_video, args['roi'][0], int(frame_count_thresh),int(skip_num))

    res_json = {
        "algorithm_data": {
            "is_alert": cnt > 0,
            "alert_time_ms": args['time_thresh'],
            "target_count": cnt,
            "target_info": []
        },
        "model_data": {
            "objects": []
        }
    }

    return json.dumps(res_json)


if __name__ == '__main__':

    p = "/project/ev_sdk/src/yolov5_deepsort/person_street.mp4"
    
    import os
    assert os.path.exists(p)#整理代码的时候加的，还没有跑过，如果确定这个mp4存在可以注释本行
    
    d = "{'roi': ['POLYGON((0.00655 0.15769),(0.27074 0.17308),(0.25983 0.98462),(0.00655 0.98846))'], 'time_thresh': 3, 'fps': 25.015745869737245}"
    handel = init()
    import gc
    import timeit

    run_times = 10 #测试调用多少次接口
    print(timeit.timeit("print(process_video(handel,p,d))",setup='gc.enable()', number=run_times, globals=locals()))
