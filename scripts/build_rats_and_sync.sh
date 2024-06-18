echo -e "\e[32mBuilding RATS\e[0m"
docker exec -u 0 rats-app-${USER} sh -c "cd /work/rats && pip install -e ."
docker exec -u 0 rats-app-${USER} sh -c "chown -R $(id -u):$(id -g) /work/rats"

libpath=$(docker exec rats-app-${USER} find /work/rats -name "*.so")
stem=$(basename ${libpath})
echo -e "\e[32mDocker library path:\e[0m $libpath"
mkdir -p /var/data/${USER}/rats/build

docker cp rats-app-${USER}:${libpath} /var/data/${USER}/rats/build/${stem}

python3 raylite sync /var/data/${USER}/rats/ray_launch.yaml
python3 raylite wcmd /var/data/${USER}/rats/ray_launch.yaml --cmd "docker cp /var/data/${USER}/rats/build/${stem} rats-app-${USER}:${libpath}"

rm -r /var/data/${USER}/rats/build
python3 raylite wcmd /var/data/${USER}/rats/ray_launch.yaml --cmd "rm -r /var/data/${USER}/rats/build"
