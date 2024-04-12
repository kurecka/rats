rm -rf docker_build
mkdir -p docker_build
cd docker_build && \
git clone git@gitlab.fi.muni.cz:risk-aware-decision-making/rats.git -b exact_lp
docker cp docker_build/rats rats-app:/work/rats
docker exec rats-app sh -c "cd /work/rats && export CMAKE_COMMON_VARIABLES=-DPEDANTIC=OFF && pip install ."
rm -rf docker_build