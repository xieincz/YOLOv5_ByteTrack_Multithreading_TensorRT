echo "install cmake 3.24.3 on home"
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev

cd /home

echo "You can also download the file from the official website"
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-45129-files/496f99a0-2e7e-4454-8bad-f90dad285655/cmake-3.24.3.tar.gz

tar -zxvf cmake-3.24.3.tar.gz

cd cmake-3.24.3/
./bootstrap

make
sudo make install
hash -r

cmake --version

echo "install opencv 4.1.2 on home"
cd /home

echo "You can also download the file from the official website"
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-45129-files/47d66834-779b-41d6-8393-8f6137d094a2/opencv412.zip
unzip opencv412.zip

find ./ -type f |xargs touch
cd opencv-4.1.2 && rm -rf build && mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 -D BUILD_opencv_python3=OFF -D BUILD_opencv_python2=OFF -D PYTHON3_EXCUTABLE=/usr/bin/python3 -D WITH_CUDA=OFF -D OPENCV_GENERATE_PKGCONFIG=ON ..

make -j8

sudo make install

pkg-config --modversion opencv4

echo "install TensorRT 7.2.2.3 on home"
sudo apt update
sudo apt install -y nano

cd /home
echo "be careful! The package installed here is only applicable to Ubuntu 18.04 cuda 11.1 & 11.2"
echo "You can also download the file from the official website"

wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-45129-files/594bdc1f-557d-40da-bf48-645e779b6808/TensorRT7.2.2.3Ubuntu1804cuda11.1cudnn8.0.tar.gz

tar -xzvf TensorRT7.2.2.3Ubuntu1804cuda11.1cudnn8.0.tar.gz

sudo echo -e "\nexport TENSORRT_ROOT=/home/TensorRT-7.2.2.3\n
export LD_LIBRARY_PATH=/home/TensorRT-7.2.2.3/lib:$LD_LIBRARY_PATH\n
export CUDA_INSTALL_DIR=/usr/local/cuda-11\n
export CUDNN_INSTALL_DIR=/usr/local/cuda-11" >> ~/.bashrc

source ~/.bashrc

sudo echo -e "\nexport TENSORRT_ROOT=/home/TensorRT-7.2.2.3\n
export LD_LIBRARY_PATH=/home/TensorRT-7.2.2.3/lib:$LD_LIBRARY_PATH\n
export CUDA_INSTALL_DIR=/usr/local/cuda-11\n
export CUDNN_INSTALL_DIR=/usr/local/cuda-11" >> /etc/profile

source /etc/profile

cd /home/TensorRT-7.2.2.3/python/
sudo pip3 install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl
cd ..
pip install uff/uff-0.6.9-py2.py3-none-any.whl
pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl

echo "install other packages"
sudo apt update
sudo apt-get install -y libeigen3-dev swig

echo "Compile the code of this project"
cd /content/YOLOv5_ByteTrack_Multithreading_TensorRT

cd yolov5_cpp_6
rm -rf build
mkdir build
cd build
cmake ..
make -j8
cd ..
cd ..

cd yolobytedxc2_6
cmake ./ \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

make -j8