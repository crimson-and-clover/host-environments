#!/usr/bin/env bash

# run as root
USER_NAME="developer"

ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh"

function run_as_root() {
    USER_ID=$1
    GROUP_ID=$2
    
    if [[ "$USER_ID" == "" ]]; then
        echo "Need at least 1 argument"
        exit 1
    fi
    
    if [[ "$GROUP_ID" == "" ]]; then
        GROUP_ID="$USER_ID"
    fi
    
    # init user
    groupadd -g "$GROUP_ID" "$USER_NAME" || exit 1
    useradd -u "$USER_ID" -g "$GROUP_ID" -m -N "$USER_NAME" -s /bin/bash || exit 1
    usermod -aG sudo "$USER_NAME" || exit 1
    cp -a /etc/skel/. "/home/$USER_NAME" || exit 1
    echo "$USER_NAME:000000" | chpasswd
    chmod 700 "/home/$USER_NAME" || exit 1
    chmod 700 "/home/$USER_NAME/hdd" || exit 1
    chmod 700 "/home/$USER_NAME/ssd" || exit 1
    chown -R "$USER_NAME:$USER_NAME" "/home/$USER_NAME" || exit 1
    
    # run as user
    su "$USER_NAME" -c "bash -v $0" || exit 1
}

function run_as_user() {
    cd $HOME
    
    # install anaconda
    mkdir -p ./downloads || exit 1
    wget -O ./downloads/Anaconda3-Linux-x86_64.sh "$ANACONDA_URL" || exit 1
    bash ./downloads/Anaconda3-Linux-x86_64.sh -b -p "$HOME/anaconda3" || exit 1
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
    run_as_user "$@"
else
    run_as_root "$@"
fi
