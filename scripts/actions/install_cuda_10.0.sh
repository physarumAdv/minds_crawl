#!/bin/bash
set -e

UBUNTU_VERSION="$(lsb_release -sr)"
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

if [[ "$UBUNTU_VERSION" == "2004" ]]; then
    echo "Cuda 10.0 does not support Ubuntu 20.04"
    exit 1
fi

wget -q "https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu$UBUNTU_VERSION-10-0-local-10.0.130-410.48_1.0-1_amd64" -O cuda-repo.deb
sudo dpkg -i cuda-repo.deb
sudo apt-key add /var/cuda-repo-*/7fa2af80.pub
sudo apt update
sudo apt -y install cuda-{compiler,libraries{,-dev}}-10-0

/usr/local/cuda-10.0/bin/nvcc -V
