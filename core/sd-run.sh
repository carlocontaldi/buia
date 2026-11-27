#!/bin/sh
# Runs sd-server and restarts it when the "active model" file changes. sd-server loads one model at start and cannot switch,
# so the Models panel (agent/sdmodels.py) writes the flags of the model you picked to that file and this script does the restart.
# usage: sd-run.sh <active-file> [default flags...]   (docker-compose.yml passes SD_ARGS / SD_VIDEO_ARGS from .env as the defaults)
# The file wins when it exists and is not empty; delete it to go back to the defaults. With neither, it waits for a model to be picked.
# It also waits while a model file the flags name (an absolute path ending in .gguf or .safetensors) is not on disk yet, and starts the server
# by itself once it is: sd-server would exit, and Docker would restart the container over and over until the first download is done.
# sd-server's output also goes to <log dir>/<active file name>.log (active-video.args -> active-video.log), because that is the only place it
# says how far a job is (its job API has no progress): the agent reads the sampling steps from it for the Video panel's progress bar.
# The file starts empty at every server start and is emptied again when it passes 10 MB; without the log volume only the file is lost.
active=$1; shift
server=${SD_SERVER:-/sd-server}
log=${SD_LOG_DIR:-/logs}/$(basename "$active" .args).log
fifo=
pid=
n=0
waiting=

stamp() { stat -c %y "$active" 2>/dev/null; }   # nanosecond modification time; empty when the file does not exist
stop() { [ -n "$pid" ] && kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; [ -n "$fifo" ] && rm -f "$fifo"; fifo=; }
trap 'stop; exit 143' TERM INT

# run <flags...>: starts sd-server in the background (pid = the server's) with its output copied by tee to the container log and to $log.
# A named pipe, not "server | tee", so that $! is the server and kill reaches it. One pipe per start: an old tee still draining the last one
# must not share its pipe with the next. If the pipe cannot be made, the server runs without the copy.
run() {
  n=$((n + 1)); fifo=/tmp/sd-out.$n; rm -f "$fifo"   # rm: a pipe left over from a container restart would make mkfifo fail
  if mkdir -p "$(dirname "$log")" 2>/dev/null && : > "$log" 2>/dev/null && mkfifo "$fifo" 2>/dev/null; then
    tee -a "$log" < "$fifo" &
    "$server" --listen-ip 0.0.0.0 --listen-port 1234 -v "$@" > "$fifo" 2>&1 &
  else
    fifo=
    "$server" --listen-ip 0.0.0.0 --listen-port 1234 -v "$@" &
  fi
  pid=$!
}

# missing <flags...>: prints the first model file the flags name that is not on disk (nothing when all are there)
missing() { for a in "$@"; do case $a in /*.gguf|/*.safetensors) [ -e "$a" ] || { echo "$a"; return; } ;; esac; done; }

# start <what> <flags...>: runs sd-server unless a model file is missing (then says so, once per file, and returns 1)
start() {
  what=$1; shift
  gone=$(missing "$@")
  if [ -n "$gone" ]; then
    [ "$waiting" = "$gone" ] || echo "sd-run: waiting for $gone (download it in Settings); not starting with $what yet"
    waiting=$gone
    return 1
  fi
  waiting=
  echo "sd-run: starting with $what"
  run "$@"
}

while :; do
  seen=$(stamp)
  if [ -s "$active" ]; then
    start "the flags in $active" $(cat "$active") || { sleep 2; continue; }
  elif [ $# -gt 0 ]; then
    start "the default flags (SD_ARGS in .env)" "$@" || { sleep 2; continue; }
  else
    sleep 2   # nothing to run yet: pick a model in the Models panel
    continue
  fi
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2
    [ -n "$fifo" ] && [ "$(stat -c %s "$log" 2>/dev/null || echo 0)" -gt 10485760 ] && : > "$log"   # tee -a keeps appending at the new end
    if [ "$(stamp)" != "$seen" ]; then
      echo "sd-run: $active changed, restarting"
      stop; pid=
      continue 2
    fi
  done
  wait "$pid"   # sd-server ended on its own: end too with its exit code, so Docker's restart policy applies as it always did
  code=$?
  [ -n "$fifo" ] && rm -f "$fifo"
  exit $code
done
