cmake_minimum_required(VERSION 3.13)
project(yolobyte)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(OpenCV REQUIRED)
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

#try to incream performemce
include(CheckIPOSupported)
check_ipo_supported(RESULT _IsIPOSupported)
  if(_IsIPOSupported)
  message(STATUS "Turn on INTERPROCEDURAL_OPTIMIZATION")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

find_package(CUDA REQUIRED)
#set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-O3;-gencode;arch=compute_75;code=sm_75)

if(WIN32)
    enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include
/home/TensorRT-7.2.2.3/include
/usr/include/x86_64-linux-gnu
)
link_directories(/usr/local/cuda/lib64
/home/TensorRT-7.2.2.3/lib
/usr/lib/x86_64-linux-gnu
)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    /usr/include/eigen3
)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)

#if(UNIX)
add_definitions(-O2 -pthread)
#endif(UNIX)
#add_definitions(-O3 -pthread)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")


# ===== yolo =====
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src YOLO_SRC_DIR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# 生成链接库.so文件
cuda_add_library(yolov5_trt SHARED ${PROJECT_SOURCE_DIR}/include/yololayer.cu ${PROJECT_SOURCE_DIR}/src/yolov5_lib.cpp)
target_link_libraries(yolov5_trt ${CUDA_LIBS} ${OpenCV_LIBS} nvonnxparser nvinfer cudart )


set(CMAKE_CXX_FLAGS "-std=c++0x")
add_definitions(-O2 -pthread)

# ===== yolobyteapi =====
SET(MODULE_NAME yolobyteapi) # 设置生成的python包的名字

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR) # 获取当前目录下的所有源文件(例如.cpp, 但是不包括.i等), 并将结果存入SRC_DIR变量中
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "")

# 设置每次更改头文件都重新生成swig接口文件
set_source_files_properties(${MODULE_NAME} PROPERTIES SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)

# 新版本的生成python包的命令
set(SRC ${PROJECT_SOURCE_DIR}/src/${MODULE_NAME}.i)
set_source_files_properties(${SRC} PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(${SRC} PROPERTIES SWIG_FLAGS "-includeall")
swig_add_library(${MODULE_NAME} LANGUAGE python SOURCES ${SRC} ${SRC_DIR} )

# target_link_libraries(${MODULE_NAME} nvinfer cudart yolov5_trt)#如果这个不行可以用下面那行
swig_link_libraries(${MODULE_NAME} nvinfer cudart yolov5_trt)

# 编译命令
# cmake ./ -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# 以及
# make
