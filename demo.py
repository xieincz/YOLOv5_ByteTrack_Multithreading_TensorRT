from yolobytedxc2_6 import yolobyteapi
import os

def init():
    # for yolo
    yolo_engine_path = '/content/YOLOv5_ByteTrack_Multithreading_TensorRT/cppmodels/yolov5m6.engine'
    
    conf_thresh = 0.5
    nms_thresh = 0.52

    # for bytetrack
    frame_rate = 30
    track_buffer = 30
    track_thresh = 0.5
    high_thresh = 0.65
    match_thresh = 0.85
    model = yolobyteapi.YoloByteAPI(yolo_engine_path, conf_thresh, frame_rate,track_buffer, track_thresh, high_thresh, match_thresh, nms_thresh)

    return model


def process_video(handle=None, input_video:str=None, **kwargs):
    output_dir='/content/outputdir'
    output_file_name='output.mp4'
    #每获取1帧要跳过几帧
    skip_num=1

    model = handle
    if os.path.exists(output_dir):
        os.system('rm -rf %s'%output_dir)
    os.mkdir(output_dir)
    res = model.processVideo(input_video,output_dir,output_file_name,int(skip_num))

    return res


if __name__ == '__main__':

    p = "/content/YOLOv5_ByteTrack_Multithreading_TensorRT/testdata/person_street.mp4"
    
    import os
    assert os.path.exists(p)#整理代码的时候加的，还没有跑过，如果确定这个mp4存在可以注释本行
    
    handel = init()
    import gc #for gc.enable()
    import timeit

    run_times = 1 #测试调用多少次接口
    print(timeit.timeit("print(process_video(handel,p))",setup='gc.enable()', number=run_times, globals=locals()))
