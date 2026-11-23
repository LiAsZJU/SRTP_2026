#!/usr/bin/env bash
# 打印 v4 四模型进度：model:epochs [exit=N]
S="${1:?usage: check_v4_progress.sh <suite_dir>}"
for d in "$S"/swin "$S"/cnn "$S"/unet "$S"/vit; do
  n=$(basename "$d")
  if [ -f "$d/metrics.jsonl" ]; then
    c=$(wc -l < "$d/metrics.jsonl")
  else
    c=0
  fi
  ec=""
  if [ -f "$S/$n.exitcode" ]; then ec=" exit=$(cat "$S/$n.exitcode")"; fi
  echo "$n:$c$ec"
done
echo "progress_done=$(ls -1 "$S"/*.exitcode 2>/dev/null | wc -l)"
