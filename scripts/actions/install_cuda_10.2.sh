#!/bin/bash
set -e

UBUNTU_VERSION="$(lsb_release -sr)"
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

if [[ "$UBUNTU_VERSION" == "2004" ]]; then
    UBUNTU_VERSION="1804"
fi

wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$UBUNTU_VERSION/x86_64/cuda-ubuntu$UBUNTU_VERSION.pin" -O cuda.pin
sudo mv cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -q "http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu$UBUNTU_VERSION-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb" -O cuda-repo.deb
sudo dpkg -i cuda-repo.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt update
sudo apt -y install cuda-{compiler,libraries{,-dev}}-10-2

/usr/local/cuda-10.2/bin/nvcc -V
