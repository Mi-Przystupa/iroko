#!/bin/bash
sudo -H pip install pip --upgrade
sudo -H pip install --upgrade numpy 
sudo -H pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl C
sudo -H pip install torchvision
sudo apt-get install bwm-ng
make -C Iroko/cluster_loadgen