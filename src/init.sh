#!/usr/bin/env bash

echo "Booting..."

# export DISPLAY=":99"

# rm /tmp/.X99-lock

# Xvfb "$DISPLAY" -screen 0 1920x1080x30 &

# echo 'DISPLAY=":99"' >> /etc/environment

# TODO set environment variables

export EGL_PLATFORM=surfaceless

echo 'EGL_PLATFORM="surfaceless"' >> /etc/environment

service ssh start

exec pause
