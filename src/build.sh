#!/usr/bin/env bash

APT_PKGS=(
    # tools
    openssh-server
    build-essential
    sudo
    git
    cmake
    curl
    wget
    net-tools
    vim
    zip
    ffmpeg
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
)

ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh"

USER_NAME="developer"

function run_as_root() {
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
    
    # create developer user
    groupadd -g 1000 "$USER_NAME" || exit 1
    useradd -u 1000 -g 1000 -m -N "$USER_NAME" || exit 1
    usermod -aG sudo "$USER_NAME" || exit 1
    echo "$USER_NAME:000000" | chpasswd
    
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
    
    # run as user
    su "$USER_NAME" -c "bash -v $0" || exit 1
}

function run_as_user() {
    cd $HOME
    
    # install anaconda
    mkdir -p ./Downloads || exit 1
    wget -O ./Downloads/Anaconda3-Linux-x86_64.sh "$ANACONDA_URL" || exit 1
    bash ./Downloads/Anaconda3-Linux-x86_64.sh -b -p "$HOME/anaconda3" || exit 1
    "$HOME/anaconda3/bin/conda" init || exit 1
    
    # set authorized keys
    mkdir -p ./.ssh && chmod -R 700 ./.ssh || exit 1
    cp -a /src/authorized_keys ./.ssh/ && \
    chmod 644 ./.ssh/authorized_keys && \
    chown "$USER_NAME:$USER_NAME" ./.ssh/authorized_keys || \
    exit 1
    
    # save cuda environment to file
    bash /src/create_cuda_env.sh > ./.cuda_env
    echo "source ~/.cuda_env" >> ~/.bashrc
}

if [[ "$USER" == "$USER_NAME" ]]; then
    run_as_user
else
    run_as_root
fi
