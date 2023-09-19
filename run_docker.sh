# Build docker: docker build -t pedenv:latest .
# Run docker:
docker run --gpus all --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 --device /dev/nvidia-modeset:/dev/nvidia-modeset --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --device /dev/nvidiactl:/dev/nvinvidiactl --ipc=host -it -v ./:/workspace/ pedenv:latest