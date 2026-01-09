#!/usr/bin/env bash
set -euo pipefail

function change_dir_permission() {
    chmod 700 "/home/$USER_NAME/" || exit 1
    chmod 700 "/home/$USER_NAME/hdd"* || exit 1
    chmod 700 "/home/$USER_NAME/ssd"* || exit 1
    chown "$USER_ID:$GROUP_ID" "/home/$USER_NAME/" || exit 1
    chown "$USER_ID:$GROUP_ID" "/home/$USER_NAME/"* || exit 1
    chown "$USER_ID:$GROUP_ID" "/home/$USER_NAME/".[^.]* || exit 1
    chown "$USER_ID:$GROUP_ID" "/home/$USER_NAME/".??* || exit 1
}

function create_user() {
    if id "$USER_NAME" &>/dev/null; then
        echo "User $USER_NAME already exists"
        return 0
    fi
    # init user
    echo "Creating user $USER_NAME"
    groupadd -g "$GROUP_ID" "$USER_NAME" || exit 1
    useradd -u "$USER_ID" -g "$GROUP_ID" -m -N "$USER_NAME" -s /bin/bash || exit 1
    usermod -aG sudo "$USER_NAME" || exit 1
    if [[ ! -e "/home/$USER_NAME/.bashrc" ]]; then
        echo "Copy skel to user home"
        cp -a /etc/skel/. "/home/$USER_NAME" || exit 1
    fi
    # Set password from environment variable or use default
    # WARNING: Default password should be changed after first login
    ROOT_PASSWORD="${ROOT_PASSWORD:-000000}"
    USER_PASSWORD="${USER_PASSWORD:-000000}"
    echo "root:${ROOT_PASSWORD}" | chpasswd
    echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd
    change_dir_permission
}

function run_as_root() {
    USER_NAME="$1"
    USER_ID="$2"
    GROUP_ID="$3"
    
    if [[ "$USER_ID" == "" ]]; then
        echo "Need at least 1 argument"
        exit 1
    fi
    
    if [[ "$GROUP_ID" == "" ]]; then
        GROUP_ID="$USER_ID"
    fi
    
    change_dir_permission
    
    # init user
    create_user
    
    change_dir_permission
    
    # run as user
    su "$USER_NAME" -c "bash -x \"$0\" \"$USER_NAME\" \"$USER_ID\" \"$GROUP_ID\"" || exit 1
}

function run_as_user() {
    cd "$HOME"
    
    # set authorized keys
    if [[ ! -d "./.ssh" ]]; then
        echo "Setting authorized keys"
        mkdir -p ./.ssh && chmod -R 700 ./.ssh || exit 1
        cp -a /src/authorized_keys ./.ssh/ && \
        chmod 644 ./.ssh/authorized_keys && \
        chown "$USER_NAME:$USER_NAME" ./.ssh/authorized_keys || exit 1
    fi
    
    # save cuda environment to file
    if [[ ! -f "./.cuda_env" ]]; then
        echo "Saving cuda environment to file"
        bash /src/create_cuda_env.sh > ./.cuda_env
        echo "source ~/.cuda_env" >> ~/.bashrc
    fi
}

USER_NAME="$1"
USER_ID="$2"
GROUP_ID="$3"

if [[ "$USER" == "$USER_NAME" ]]; then
    run_as_user "$@"
else
    run_as_root "$@"
fi
