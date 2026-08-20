#!/bin/bash
set -e

echo "=================================================="
echo " LiteTorch C++ Dependencies Auto-Installer"
echo "=================================================="

if command -v apt-get >/dev/null 2>&1; then
    echo "Detected Debian/Ubuntu system. Installing dependencies via apt..."
    sudo apt-get update
    sudo apt-get install -y build-essential g++ make ocl-icd-opencl-dev opencl-headers python3-dev python3-pip
elif command -v dnf >/dev/null 2>&1; then
    echo "Detected RHEL/Fedora system. Installing dependencies via dnf..."
    sudo dnf install -y gcc-c++ make ocl-icd-devel opencl-headers python3-devel python3-pip
elif command -v pacman >/dev/null 2>&1; then
    echo "Detected Arch Linux system. Installing dependencies via pacman..."
    sudo pacman -S --needed base-devel gcc make opencl-headers ocl-icd python-pip
else
    echo "Unknown package manager. Please install g++, make, opencl-headers, and python3-dev manually."
    exit 1
fi

echo "=================================================="
echo " C++ Dependencies Installed Successfully!"
echo "=================================================="
