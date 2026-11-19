#!/bin/bash
# Usage: bench.sh <outdir> <models dir> <label>=<image> ...; on the card's Docker host, with the card idle.
# Every file runs on every image in the order A B C C B A: llama-bench, then llama-server with MTP and client.py.
set -u
USAGE="usage: bench.sh <outdir> <models dir> <label>=<image> ..."
OUT=${1:?$USAGE}
MODELS=$(realpath "${2:?$USAGE}") || exit 2
shift 2
[ $# -ge 1 ] || { echo "$USAGE"; exit 2; }
FILES=${FILES:-Q4_1 Q4_K_M Q8_0 UD-IQ3_XXS UD-Q4_K_XL UD-Q6_K}
PREFIX=${PREFIX:-Qwen3.8-27B-}
REPS=${REPS:-5} CTX=${CTX:-32768} START=${START:-50} CAP=${CAP:-} PORT=${PORT:-8095}
MAX_CHIP=${MAX_CHIP:-95} MAX_MEMORY=${MAX_MEMORY:-85}
BENCH_ARGS=${BENCH_ARGS:--ngl 99 -fa 1 -lm dio -b 2048 -ub 2048 -p 4,8,16,512,2048 -n 128}
SERVER_ARGS=${SERVER_ARGS:--ngl 99 -fa on -lm dio -c $CTX -np 4 --kv-unified -b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 --jinja}
NAME=gfx906-bench
HERE=$(cd "$(dirname "$0")" && pwd)
# 0x66a0-0x66a7 and 0x66af are the Vega 20 boards: MI50, MI60, Radeon Pro VII, Radeon VII.
DEV=${CARD:-$(dirname "$(grep -l -x -E '0x66a[0-7f]' /sys/class/drm/card*/device/device 2>/dev/null | head -1)")}
[ -r "$DEV/mem_info_vram_used" ] || { echo "no gfx906 card found, set CARD=/sys/class/drm/cardN/device"; exit 2; }
H=$(ls -d "$DEV"/hwmon/hwmon* | head -1)
milli() { local v; v=$(cat "$H/$1" 2>/dev/null) || v=0; echo $(( v / ${2:-1000} )); }
vram() { echo $(( $(cat "$DEV/mem_info_vram_used") / 1048576 )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/bench.log"; }
trap 'docker rm -f $NAME >/dev/null 2>&1' EXIT

declare -a LABELS IMAGES
for a in "$@"; do
  [[ $a == *=* ]] || { echo "$USAGE"; exit 2; }
  LABELS+=("${a%%=*}") IMAGES+=("${a#*=}")
done
mkdir -p "$OUT"
[ "$(vram)" -lt 1024 ] || { echo "the card is in use ($(vram) MiB), stop its server first"; exit 2; }

# Pulled first, so that no run waits on the network; the digests make the run repeatable.
{
  echo "{\"date\": \"$(date -u +%FT%TZ)\", \"cap\": \"$CAP\", \"reps\": $REPS, \"ctx\": $CTX,"
  echo " \"bench_args\": \"$BENCH_ARGS\", \"server_args\": \"$SERVER_ARGS\", \"images\": ["
  for i in "${!LABELS[@]}"; do
    docker pull -q "${IMAGES[$i]}" >/dev/null || { echo "cannot pull ${IMAGES[$i]}" >&2; exit 2; }
    digest=$(docker image inspect -f '{{index .RepoDigests 0}}' "${IMAGES[$i]}" 2>/dev/null)
    version=$(docker run --rm --entrypoint /app/llama-server "${IMAGES[$i]}" --version 2>&1 | grep -m1 '^version' | tr -d '"')
    [ "$i" -gt 0 ] && echo ","
    echo "  {\"label\": \"${LABELS[$i]}\", \"image\": \"${IMAGES[$i]}\", \"digest\": \"$digest\", \"version\": \"$version\"}"
  done
  echo "]}"
} > "$OUT/meta.json" || exit 2

# Waits until the chip is back at START, so that every run begins equally cool; ten minutes at most.
cool() {
  local waited=0
  until [ "$(milli temp2_input)" -le "$START" ] || [ $waited -ge 600 ]; do sleep 5; waited=$((waited + 5)); done
}

# One line a second while the container runs; stops it at the temperature limits and flags a power cap other than CAP.
watch() {
  local run=$1 kind=$2 chip mem cap
  : > "$run.$kind.samples"
  while [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ]; do
    chip=$(milli temp2_input) mem=$(milli temp3_input) cap=$(milli power1_cap 1000000)
    echo "$(date +%s) chip=$chip memory=$mem draw=$(milli power1_input 1000000) cap=$cap sclk=$(milli freq1_input 1000000) mclk=$(milli freq2_input 1000000)" >> "$run.$kind.samples"
    if [ "$chip" -ge "$MAX_CHIP" ] || [ "$mem" -ge "$MAX_MEMORY" ]; then
      echo "stopped at $chip °C chip, $mem °C memory" >> "$run.flags"; docker kill $NAME >/dev/null 2>&1
    fi
    if [ -n "$CAP" ] && [ "$cap" -ne "$CAP" ]; then echo "power cap $cap W, not $CAP W" >> "$run.flags"; fi
    sleep 1
  done
}

bench() {
  local image=$1 file=$2 run=$3
  docker rm -f $NAME >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name $NAME --device /dev/kfd --device /dev/dri -v "$MODELS":/models:ro --entrypoint /app/llama-bench \
    "$image" -m "/models/$file" $BENCH_ARGS -r "$REPS" -o jsonl >/dev/null || { echo "did not start" >> "$run.flags"; return; }
  watch "$run" bench
  docker logs $NAME 2>/dev/null > "$run.bench.jsonl"
  docker logs $NAME 2>&1 >/dev/null | tail -20 > "$run.bench.log"
  [ "$(docker inspect -f '{{.State.ExitCode}}' $NAME)" = 0 ] || echo "llama-bench exited $(docker inspect -f '{{.State.ExitCode}}' $NAME)" >> "$run.flags"
  docker rm -f $NAME >/dev/null 2>&1
}

serve() {
  local image=$1 file=$2 run=$3 i
  docker rm -f $NAME >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name $NAME --device /dev/kfd --device /dev/dri -v "$MODELS":/models:ro -p "127.0.0.1:$PORT:8080" \
    --entrypoint /app/llama-server "$image" -m "/models/$file" $SERVER_ARGS --host 0.0.0.0 --port 8080 >/dev/null \
    || { echo "did not start" >> "$run.flags"; return; }
  watch "$run" server &
  local watcher=$!
  for i in $(seq 600); do
    [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
    [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || break
    sleep 1
  done
  python3 "$HERE/client.py" "http://127.0.0.1:$PORT" "$run.server.json" >> "$OUT/bench.log" 2>&1 \
    || echo "the client failed" >> "$run.flags"
  docker logs $NAME 2>&1 | tail -20 > "$run.server.log"
  docker rm -f $NAME >/dev/null 2>&1
  wait $watcher
}

ORDER=("${!LABELS[@]}")
for ((i = ${#LABELS[@]} - 1; i >= 0; i--)); do ORDER+=("$i"); done
for q in $FILES; do
  f=$PREFIX$q.gguf
  [ -f "$MODELS/$f" ] || { say "missing $f"; continue; }
  mkdir -p "$OUT/$q"
  pass=0
  for i in "${ORDER[@]}"; do
    pass=$((pass + 1))
    run="$OUT/$q/${LABELS[$i]}.$pass"
    cool; say "$q ${LABELS[$i]} llama-bench"; bench "${IMAGES[$i]}" "$f" "$run"
    cool; say "$q ${LABELS[$i]} server"; serve "${IMAGES[$i]}" "$f" "$run"
    [ -s "$run.flags" ] && say "$q ${LABELS[$i]}: $(sort -u "$run.flags" | tr '\n' ' ')"
  done
done
say "done"
