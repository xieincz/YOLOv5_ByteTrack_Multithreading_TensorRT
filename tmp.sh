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
export CUDA_INSTALL_DIR=/usr/local/cuda-11.1\n
export CUDNN_INSTALL_DIR=/usr/local/cuda-11.1" >> ~/.bashrc

source ~/.bashrc

sudo echo -e "\nexport TENSORRT_ROOT=/home/TensorRT-7.2.2.3\n
export LD_LIBRARY_PATH=/home/TensorRT-7.2.2.3/lib:$LD_LIBRARY_PATH\n
export CUDA_INSTALL_DIR=/usr/local/cuda-11.1\n
export CUDNN_INSTALL_DIR=/usr/local/cuda-11.1" >> /etc/profile

source /etc/profile

cd /home/TensorRT-7.2.2.3/python/
sudo pip3 install tensorrt-7.2.2.3-cp37-none-linux_x86_64.whl
cd ..
pip install uff/uff-0.6.9-py2.py3-none-any.whl
pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
