docker exec -u 1000:100 ray_container sh -c "cd /work/rats && export CMAKE_COMMON_VARIABLES=-DPEDANTIC=OFF && pip install ."
docker cp ray_container:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so /var/data/xkurecka/rats/build/rats.so
python3 pyrats/raylite sync ray_launch.yaml
python3 pyrats/raylite cmd ray_launch.yaml "docker cp /var/data/xkurecka/rats/build/rats.so ray_container:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so"