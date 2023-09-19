# Run carla in Docker: 
# https://carla.readthedocs.io/en/0.9.14/build_docker/

docker pull carlasim/carla:0.9.14

# Use this to start the carla server:
docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen

# IMPORTANT:
# Restart the server before starting the recodring of the test sequences.
# Somethimes it happens that pedestrians arent removed properly from the 
# simulation and they are walking around in the started simulation but no skeleton is delivered.