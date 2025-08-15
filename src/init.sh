#!/usr/bin/env bash

echo "Booting..."

export EGL_PLATFORM=surfaceless
export LANG="en_US.UTF-8"
export LANGUAGE="en_US:en"
export LC_ALL="en_US.UTF-8"
export TZ="Asia/Shanghai"

# Function to add environment variable if not already present
function add_env_var() {
    local var_name="$1"
    local var_value="$2"
    local env_line="${var_name}=\"${var_value}\""
    
    # Check if the variable is already set in /etc/environment
    if ! grep -q "^${var_name}=" /etc/environment 2>/dev/null; then
        echo "Adding ${var_name} to /etc/environment"
        echo "${env_line}" >> /etc/environment
    else
        echo "${var_name} already exists in /etc/environment"
    fi
}

# Add environment variables only if they don't exist
add_env_var "EGL_PLATFORM" "$EGL_PLATFORM"
add_env_var "LANG" "$LANG"
add_env_var "LANGUAGE" "$LANGUAGE"
add_env_var "LC_ALL" "$LC_ALL"
add_env_var "TZ" "$TZ"

service ssh start

exec pause
