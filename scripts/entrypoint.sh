#!/bin/sh
# Container entry point.
#   serve                   run the API on $PORT (the default command)
#   train [options]         train, evaluate, and save the model (python -m ml_pipeline.train),
#                           e.g. train --set model.type=dummy
#   anything else           run it as a command, e.g. python -m data_pipeline.push
set -eu

command="${1:-serve}"
if [ "$#" -gt 0 ]; then shift; fi

case "$command" in
  serve) exec uvicorn api.main:create_app --factory --host 0.0.0.0 --port "${PORT:-8080}" "$@" ;;
  train) exec python -m ml_pipeline.train --config "${CONFIG:-config.yaml}" "$@" ;;
  *) exec "$command" "$@" ;;
esac
