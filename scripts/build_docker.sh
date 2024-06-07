# NAMESPACE="gitlab.fi.muni.cz:5050/risk-aware-decision-making"
NAMESPACE=xkurecka
TAG='latest'
BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)

GIT_IMAGE_NAME=$NAMESPACE/rats:$COMMIT
TAG_IMAGE_NAME=$NAMESPACE/rats:$TAG

# Use blue color
echo -e "\033[0;34mBuilding docker image for RATS\033[0m"
echo
if [ -z "$TAG" ]; then
    echo "No tag specified. Using commit hash as tag."
    echo "Docker image name with tag: $GIT_IMAGE_NAME"
else
    echo "Tag specified: $TAG"
    echo "Using both commit hash and tag."
    echo -e "\033[0;32mDocker image name with tag:\033[0m $GIT_IMAGE_NAME"
    echo -e "\033[0;32mDocker image name with tag:\033[0m $TAG_IMAGE_NAME"
fi
# green
echo ""
echo -e "\033[0;32mGit branch:\033[0m $BRANCH"
echo -e "\033[0;32mGit commit:\033[0m $COMMIT"

# Check git is clean
if [ -n "$(git status --porcelain)" ]; then
    echo
    echo -e "\033[0;31mGit is not clean. Please commit your changes before building the docker image.\033[0m"
    exit 1
fi

# Check remote is up-to-date
if [ "$(git rev-list HEAD...origin/$BRANCH --count)" -ne 0 ]; then
    echo -e "\033[0;31mRemote is not up-to-date. Please push your changes before building the docker image.\033[0m"
    exit 1
fi

docker build --platform linux/x86_64 . -t $GIT_IMAGE_NAME
echo -e "\033[0;32mDocker image $GIT_IMAGE_NAME built successfully.\033[0m"
if [ -n "$TAG" ]; then
    echo -e "\033[0;32mDocker image $TAG_IMAGE_NAME built successfully.\033[0m"
    docker tag $GIT_IMAGE_NAME $TAG_IMAGE_NAME
fi
