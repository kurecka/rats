USER="kurecka"

mkdir -p docker_build && \
rm -rf docker_build/rats && \
cd docker_build && \
git clone git@gitlab.fi.muni.cz:risk-aware-decision-making/rats.git && \
# -b feature/pareto_uct && \
docker build --platform linux/x86_64 .. -t $USER/rats:latest
# rm -rf docker_build #&& \
#docker push $USER/rats:latest
