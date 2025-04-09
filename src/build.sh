#!/usr/bin/env bash

APT_PKGS=(
    # tools
    openssh-server
    build-essential
    sudo
    git
    cmake
    ninja-build
    curl
    wget
    net-tools
    vim
    zip
    ffmpeg
    iputils-ping
    iproute2
    # compile dependency
    libgl1-mesa-glx
    libegl1-mesa
    libxrandr2
    libxrandr2
    libxss1
    libxcursor1
    libxcomposite1
    libasound2
    libxi6
    libxtst6
    libboost-program-options-dev
    libboost-filesystem-dev
    libboost-graph-dev
    libboost-system-dev
    libeigen3-dev
    libflann-dev
    libfreeimage-dev
    libmetis-dev
    libgoogle-glog-dev
    libgtest-dev
    libgmock-dev
    libsqlite3-dev
    libglew-dev
    qtbase5-dev
    libqt5opengl5-dev
    libcgal-dev
    libceres-dev
)

USER_NAME="developer"

cd /root

chmod -R 777 /src || exit 1

# change root password
echo "root:000000" | chpasswd

# upgrade and install package
apt-get update && apt-get upgrade -y && apt-get install -y "${APT_PKGS[@]}" || exit 1

# compile entrypoint
gcc /src/pause.c -o /usr/bin/pause || exit 1

# compile cuda_occupier to deal with hpc policy
nvcc /src/cuda_occupier.cu -o /usr/bin/cuda_occupier -lnvidia-ml || exit 1

# set authorized keys
mkdir -p ./.ssh && chmod -R 700 ./.ssh || exit 1
cp -a /src/authorized_keys ./.ssh/ && \
chmod 644 ./.ssh/authorized_keys && \
chown root:root ./.ssh/authorized_keys || \
exit 1

# allow root login
sed -i '/#PermitRootLogin/s/.*/PermitRootLogin yes/' /etc/ssh/sshd_config || exit 1

# save cuda environment to file
bash /src/create_cuda_env.sh > ./.cuda_env
echo "source ~/.cuda_env" >> ~/.bashrc
