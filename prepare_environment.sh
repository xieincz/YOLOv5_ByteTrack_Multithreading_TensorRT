sudo apt-get update
sudo apt-get install -y build-essential libssl-dev

cd /home

wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-45129-files/496f99a0-2e7e-4454-8bad-f90dad285655/cmake-3.24.3.tar.gz

tar -zxvf cmake-3.24.3.tar.gz

cd cmake-3.24.3/
./bootstrap

make
sudo make install
hash -r

cmake --version
