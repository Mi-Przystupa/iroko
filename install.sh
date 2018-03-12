#!/bin/bash

# fetch and install mininet
sudo git submodule update --init --recursive
cd mininet
sudo mininet/util/install.sh -nfv
cd ..
sudo -H pip install pip --upgrade
# required for reinforcement learning
sudo -H pip install --upgrade numpy 
sudo -H pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
sudo -H pip install torchvision
# for generating plots
sudo -H pip install matplotlib
# traffic monitor
sudo apt-get install -y bwm-ng
# traffic generator
make -C iroko/cluster_loadgen
# reset accidental root files to the original user
sudo chown -R $USER:$USER .

