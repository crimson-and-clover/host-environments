#!/usr/bin/env bash

# run as root
USER_NAME="developer"

NEW_USER_ID=$1
NEW_GROUP_ID=$2

if [[ "$NEW_USER_ID" == "" ]]; then
    echo "Need at least 1 argument"
    exit 1
fi

if [[ "$NEW_GROUP_ID" == "" ]]; then
    NEW_GROUP_ID="$NEW_USER_ID"
fi

usermod -u "$NEW_USER_ID" "$USER_NAME" || exit 1
groupmod -g "$NEW_GROUP_ID" "$USER_NAME" || exit 1
chown -R "$USER_NAME:$USER_NAME" "/home/$USER_NAME" || exit 1
