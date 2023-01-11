#!/bin/bash

# set model size（MiB）
# --half --workspace 9 need at least 10989 MiB
min_size=15000

# use 'nvidia-smi' API to get used memory
memory=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i 0`
memory=${memory: 0: -3}

echo ${memory}

# must use two (), or system will recogize it as set min_size as the input file of memory
while ((memory<min_size))
do
    echo "Free memory ${memory}MiB is not enough, Wait to Execute..."
	sleep 5
	memory=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i 0`
    memory=${memory: 0: -3}
done

# Execute code
cd /project/ev_sdk/src/yolov5_cpp_6/build

./yolov5 -s yolov5l6.wts yolov5l6.engine l6
