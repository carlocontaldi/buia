#!/bin/sh
# Runs sd-server and restarts it when the "active model" file changes. sd-server loads one model at start and cannot switch,
# so the Models panel (agent/sdmodels.py) writes the flags of the model you picked to that file and this script does the restart.
# usage: sd-run.sh <active-file> [default flags...]   (docker-compose.yml passes SD_ARGS / SD_VIDEO_ARGS from .env as the defaults)
# The file wins when it exists and is not empty; delete it to go back to the defaults. With neither, it waits for a model to be picked.
active=$1; shift
server=${SD_SERVER:-/sd-server}
pid=

stamp() { stat -c %Y "$active" 2>/dev/null; }   # modification time; empty when the file does not exist
stop() { [ -n "$pid" ] && kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }
trap 'stop; exit 143' TERM INT

while :; do
  seen=$(stamp)
  if [ -s "$active" ]; then
    echo "sd-run: starting with the flags in $active"
    "$server" --listen-ip 0.0.0.0 --listen-port 1234 -v $(cat "$active") &
  elif [ $# -gt 0 ]; then
    echo "sd-run: starting with the default flags (SD_ARGS in .env)"
    "$server" --listen-ip 0.0.0.0 --listen-port 1234 -v "$@" &
  else
    sleep 2   # nothing to run yet: pick a model in the Models panel
    continue
  fi
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2
    if [ "$(stamp)" != "$seen" ]; then
      echo "sd-run: $active changed, restarting"
      stop; pid=
      continue 2
    fi
  done
  wait "$pid"   # sd-server ended on its own: end too with its exit code, so Docker's restart policy applies as it always did
  exit $?
done
