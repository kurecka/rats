docker exec -u 1000:100 ray_container sh -c "cd /work/rats && export CMAKE_COMMON_VARIABLES=-DPEDANTIC=OFF && pip install ."
# docker cp ray_container:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so /var/data/${USER}/rats/build/rats.so
# raylite sync /var/data/${USER}/rats/ray_launch.yaml
# raylite cmd /var/data/${USER}/rats/ray_launch.yaml "docker cp /var/data/${USER}/rats/build/rats.so ray_container:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so"
