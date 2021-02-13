#!/bin/bash
set -e

UBUNTU_VERSION="$(lsb_release -sr)"
UBUNTU_VERSION="${UBUNTU_VERSION//./}"

wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$UBUNTU_VERSION/x86_64/cuda-ubuntu$UBUNTU_VERSION.pin" -O cuda.pin
sudo mv cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -q "http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu$UBUNTU_VERSION-11-0-local_11.0.2-450.51.05-1_amd64.deb" -O cuda-repo.deb
sudo dpkg -i cuda-repo.deb
sudo apt-key add "/var/cuda-repo-ubuntu$UBUNTU_VERSION-11-0-local/7fa2af80.pub"
sudo apt update
sudo apt -y install cuda-{compiler,libraries{,-dev}}-11-0

/usr/local/cuda-11.0/bin/nvcc -V
