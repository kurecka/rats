docker exec -u 1000:1000 rats-app-${USER} sh -c "cd /work/rats && export CMAKE_COMMON_VARIABLES=-DPEDANTIC=OFF && pip install ."
docker cp rats-app-${USER}:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so /var/data/${USER}/rats/build/rats.so
raylite sync /var/data/${USER}/rats/ray_launch.yaml
raylite wcmd /var/data/${USER}/rats/ray_launch.yaml --cmd "docker cp /var/data/${USER}/rats/build/rats.so rats-app-${USER}:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so"
