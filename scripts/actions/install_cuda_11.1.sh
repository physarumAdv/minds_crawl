#!/bin/bash
set -e

UBUNTU_VERSION="$(lsb_release -sr)"
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$UBUNTU_VERSION/x86_64/cuda-ubuntu$UBUNTU_VERSION.pin" -O cuda.pin
sudo mv cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -q "http://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu$UBUNTU_VERSION-11-1-local_11.1.0-455.23.05-1_amd64.deb" -O cuda-repo.deb
sudo dpkg -i cuda-repo.deb
sudo apt-key add "/var/cuda-repo-ubuntu$UBUNTU_VERSION-11-1-local/7fa2af80.pub"
sudo apt update
sudo apt -y install cuda-{compiler,libraries{,-dev}}-11-1

/usr/local/cuda-11.1/bin/nvcc -V
