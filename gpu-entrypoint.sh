#!/bin/bash
USERNAME=$1

echo "=== PhantomX GPU Entrypoint ==="
chown $USERNAME:$USERNAME /home/$USERNAME

# Initialize NVIDIA GPU
echo "Initializing GPU..."
nvidia-smi

echo "Running post-init as $USERNAME..."
su -c "bash ./post-init.sh" $USERNAME
