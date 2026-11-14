#!/bin/sh
# Container entry point.
#   serve                   run the API on $PORT (the default command)
#   anything else           run it as a command, e.g. python -m data_pipeline.push
# A `train` command comes with the ML pipeline (ml_pipeline/README.md).
set -eu

command="${1:-serve}"
if [ "$#" -gt 0 ]; then shift; fi

case "$command" in
  serve) exec uvicorn api.main:create_app --factory --host 0.0.0.0 --port "${PORT:-8080}" "$@" ;;
  *) exec "$command" "$@" ;;
esac
