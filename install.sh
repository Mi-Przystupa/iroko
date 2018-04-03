#!/bin/bash

# fetch and install mininet
git submodule update --init --recursive
cd mininet
sudo mininet/util/install.sh -nfv
cd ..

# Install pip
sudo apt-get install python-pip
sudo -H pip install pip --upgrade
# required for reinforcement learning
sudo -H pip install --upgrade numpy 
sudo -H pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
sudo -H pip install torchvision
# for generating plots
sudo apt-get install python-matplotlib
# some regex magic
sudo -H pip install sre_yield
# traffic monitors
sudo apt-get install -y bwm-ng
sudo apt-get install -y ifstat
# traffic generators
sudo apt-get install -y iperf
make -C iroko/cluster_loadgen
# reset accidental root files to the original user
sudo chown -R $USER:$USER .

