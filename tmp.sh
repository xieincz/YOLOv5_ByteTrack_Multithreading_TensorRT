echo "Compile the code of this project"
cd /content/YOLOv5_ByteTrack_Multithreading_TensorRT

cd yolov5_cpp_6
rm -rf build
mkdir build
cd build
cmake ..
make
cd ..
cd ..

cd yolobytedxc2_6
cmake ./ \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

make 
