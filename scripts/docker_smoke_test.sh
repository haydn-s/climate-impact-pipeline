#!/bin/sh
# Check of the Docker image (make docker-smoke): the image builds, its
# packages import, and the data pipeline and train commands start, as a
# non-root user. It doesn't train a model or call the API, and it isn't in CI.
set -eu

IMAGE="${IMAGE:-project3}"
RUN_AS="$(id -u):$(id -g)"   # nothing runs as root

echo "--> Importing the packages in the container"
docker run --rm --user "$RUN_AS" "$IMAGE" \
  python -c "import api.main, data_pipeline.ingest, data_pipeline.prepare, data_pipeline.push, ml_pipeline"

echo "--> Starting each data pipeline command"
for stage in ingest prepare push; do
  docker run --rm --user "$RUN_AS" "$IMAGE" python -m "data_pipeline.$stage" --help >/dev/null
done

echo "--> Starting the train command"
docker run --rm --user "$RUN_AS" "$IMAGE" train --help >/dev/null
echo "--> Docker smoke test passed"
