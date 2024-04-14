docker exec -u 21009:10100 rats-app sh -c "cd /work/rats && export CMAKE_COMMON_VARIABLES=-DPEDANTIC=OFF && pip install ."
docker cp rats-app:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so /var/data/${USER}/rats/build/rats.so
pyrats/raylite sync /var/data/${USER}/rats/ray_launch.yaml
pyrats/raylite cmd /var/data/${USER}/rats/ray_launch.yaml "docker cp /var/data/${USER}/rats/build/rats.so rats-app:/home/ray/anaconda3/lib/python3.8/site-packages/rats.cpython-38-x86_64-linux-gnu.so"
