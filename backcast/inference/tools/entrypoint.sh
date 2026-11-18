#!/bin/sh
set -eu

if [ "${1:-}" = "runpod-worker" ]; then
    shift
    if [ "$#" -ne 0 ]; then
        echo "runpod-worker takes no extra arguments" >&2
        exit 64
    fi
    if [ -z "${PUBLIC_KEY:-}" ]; then
        echo "runpod-worker requires PUBLIC_KEY with an authorized SSH public key" >&2
        exit 64
    fi
    install -d -m 0700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 0600 /root/.ssh/authorized_keys
    ssh-keygen -l -f /root/.ssh/authorized_keys >/dev/null
    ssh-keygen -A
    mkdir -p /run/sshd
    exec /usr/sbin/sshd -D -e \
        -o PermitRootLogin=prohibit-password \
        -o PasswordAuthentication=no \
        -o KbdInteractiveAuthentication=no \
        -o PubkeyAuthentication=yes \
        -o UsePAM=no \
        -o SetEnv=PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
fi

exec deepstarnoodles-inference "$@"
