#!/bin/bash
set -e

UBUNTU_VERSION="$(lsb_release -sr)"
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

if [[ "$UBUNTU_VERSION" == "2004" ]]; then
    UBUNTU_VERSION="1804"
fi

wget -q "https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu$UBUNTU_VERSION-10-1-local-10.1.105-418.39_1.0-1_amd64.deb" -O cuda-repo.deb
sudo dpkg -i cuda-repo.deb
sudo apt-key add /var/cuda-repo-*/7fa2af80.pub
sudo apt update
sudo apt -y install cuda-{compiler,libraries{,-dev}}-10-1

/usr/local/cuda-10.1/bin/nvcc -V
