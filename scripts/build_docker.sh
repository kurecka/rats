NAMESPACE="gitlab.fi.muni.cz:5050/risk-aware-decision-making"
BRANCH="master"

rm -rf docker_build
mkdir -p docker_build && \
rm -rf docker_build/rats && \
cd docker_build && \
git clone git@gitlab.fi.muni.cz:risk-aware-decision-making/rats.git -b $BRANCH && \
docker build --platform linux/x86_64 .. -t $NAMESPACE/rats:latest && \
rm -rf docker_build
docker push $NAMESPACE/rats:latest
