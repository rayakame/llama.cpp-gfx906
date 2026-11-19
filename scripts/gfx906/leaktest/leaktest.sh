#!/bin/bash
# Usage: leaktest.sh <image> <model.gguf> [requests]; on the card's Docker host, with the card idle.
# Exit 0: at most LIMIT MiB per 100 requests in the second pass. Exit 1: it leaks. Exit 2: could not test.
set -u
USAGE="usage: leaktest.sh <image> <model.gguf> [requests]"
IMG=${1:?$USAGE}
MODEL=$(realpath "${2:?$USAGE}") || exit 2
N=${3:-200}
LIMIT=${LIMIT:-10} PORT=${PORT:-8099} MAX_CHIP=${MAX_CHIP:-95} MAX_MEMORY=${MAX_MEMORY:-85}
NAME=gfx906-leaktest
HERE=$(cd "$(dirname "$0")" && pwd)
# 0x66a0-0x66a7 and 0x66af are the Vega 20 boards: MI50, MI60, Radeon Pro VII, Radeon VII.
DEV=${CARD:-$(dirname "$(grep -l -x -E '0x66a[0-7f]' /sys/class/drm/card*/device/device 2>/dev/null | head -1)")}
[ -r "$DEV/mem_info_vram_used" ] || { echo "no gfx906 card found, set CARD=/sys/class/drm/cardN/device"; exit 2; }
H=$(ls -d "$DEV"/hwmon/hwmon* | head -1)
vram() { echo $(( $(cat "$DEV/mem_info_vram_used") / 1048576 )); }
temp() { echo $(( $(cat "$H/$1") / 1000 )); }
cleanup() {
  [ -n "${GUARD:-}" ] && kill "$GUARD" 2>/dev/null
  docker rm -f $NAME >/dev/null 2>&1
}
trap cleanup EXIT

[ "$(vram)" -lt 1024 ] || { echo "the card is in use ($(vram) MiB), stop its server first"; exit 2; }
docker run -d --name $NAME --device /dev/kfd --device /dev/dri -v "$(dirname "$MODEL")":/models:ro \
  -v "$HERE":/leaktest:ro -p "$PORT":8080 --entrypoint bash "$IMG" -c \
  "\$(hipconfig -l)/clang -O2 -shared -fPIC -o /tmp/hipshim.so /leaktest/hipshim.c -ldl -lpthread &&
   LD_PRELOAD=/tmp/hipshim.so exec /app/llama-server --host 0.0.0.0 --port 8080 -m '/models/$(basename "$MODEL")' \
     -c 8192 -ngl 99 -fa on -np 1" >/dev/null || exit 2

# temp2 is the junction, temp3 the memory.
( while [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ]; do
    sleep 5
    if [ "$(temp temp2_input)" -ge "$MAX_CHIP" ] || [ "$(temp temp3_input)" -ge "$MAX_MEMORY" ]; then
      echo "stopped at $(temp temp2_input) °C chip, $(temp temp3_input) °C memory"; docker kill $NAME >/dev/null
    fi
  done ) &
GUARD=$!

for i in $(seq 300); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
  [ "$i" = 300 ] && { echo "the server did not come up"; docker logs $NAME 2>&1 | tail -5; exit 2; }
  sleep 1
done

# The same requests twice: the first pass lets buffers and lazily loaded kernels reach their size, only a leak
# still grows in the second.
alive() { [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ]; }
python3 "$HERE/load.py" "http://127.0.0.1:$PORT" "$N" >/dev/null && alive || { echo "the first pass did not finish"; exit 2; }
V1=$(vram)
python3 "$HERE/load.py" "http://127.0.0.1:$PORT" "$N" >/dev/null && alive || { echo "the second pass did not finish"; exit 2; }
V2=$(vram)

SHIM=$(docker logs $NAME 2>&1 | grep '^HIPSHIM t=' | tail -1 | cut -d' ' -f3-)
PER100=$(( (V2 - V1) * 100 / N ))
echo "image $IMG"
echo "card memory after the first pass of $N requests: $V1 MiB, after the second: $V2 MiB, $PER100 MiB per 100 requests"
echo "graph calls: $SHIM"
if [ "$PER100" -le "$LIMIT" ]; then echo "PASS"; exit 0; else echo "LEAK"; exit 1; fi
