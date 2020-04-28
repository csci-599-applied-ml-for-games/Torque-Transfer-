#!/bin/sh
export CFLAGS="-fPIC"
export CPPFLAGS=$CFLAGS
export CXXFLAGS=$CFLAGS
echo "Downloading protobuf..."
echo "----------------"
wget -N https://github.com/protocolbuffers/protobuf/releases/download/v2.6.0/protobuf-2.6.0.tar.gz
echo "----------------"
echo "Extracting protobuf..."
echo "----------------"
tar -xvzf protobuf-2.6.0.tar.gz
echo "----------------"
echo "Building protobuf...."
echo "----------------"
cd protobuf-2.6.0
echo "Configuring protobuf..."
./configure
echo "Installing protobuf..."
make && make install
echo "Install finished. Leaving directory..."
cd /code
echo "----------------"
echo "Cloning torcs..."
echo "----------------"
git clone https://github.com/Skeletrox/torcs-1.3.7.git
cd torcs-1.3.7
echo "----------------"
echo "Building torcs..."
echo "----------------"
echo "Configuring torcs..."
./configure --prefix=$(pwd)/BUILD
echo "Installing torcs..."
make && make install && make datainstall
echo "Setting up screenpipe and compiling IPC command..."
cd screenpipe
g++ IPC_command.cpp torcs_data.pb.cc -o IPC_command `pkg-config --cflags --libs opencv protobuf libzmq`
echo "Running ldconfig..."
ldconfig
echo "setup complete"