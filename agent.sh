#!/usr/bin/env bash
# agent.sh -- runs on the RunPod pod. Pulls jobs from the GitHub repo, runs them
# one at a time, and pushes their logs back, so they can be read remotely.
#
#   cd /workspace/relay && DATA_DIR=/workspace/data nohup bash agent.sh > /workspace/agent.out 2>&1 &
#
# Repo layout (created if missing):
#   code/          training scripts (pulled before every job)
#   jobs/queue/    NNN_name.sh  -- waiting jobs, run in name order
#   jobs/done/     finished jobs (moved here with their exit code in the log)
#   jobs/cancel    write a job name into this file to stop that job
#   status/        <job>.log (stdout), runs/<run>/{train.log,args.json,rollout_cache.json},
#                  heartbeat.txt (time, GPU, current job)
# Jobs see: $DATA_DIR (the dataset), $RUNS (/workspace/runs, checkpoints stay
# on the pod), and run with code/ as working directory.
set -u
cd "$(dirname "$0")"
REPO=$(pwd)
RUNS=${RUNS:-/workspace/runs}
DATA_DIR=${DATA_DIR:?set DATA_DIR to the folder with sampled_states_physical.npy}
PUSH_EVERY=${PUSH_EVERY:-600}
mkdir -p jobs/queue jobs/done status/runs code "$RUNS"
git config user.name "runpod-agent"
git config user.email "agent@runpod.local"
export DATA_DIR RUNS

cur_pid=""; cur_job=""; last_push=0

sync_logs() {
  for d in "$RUNS"/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d"); mkdir -p "status/runs/$n"
    for f in train.log args.json rollout_cache.json config.json; do
      [ -f "$d$f" ] && cp "$d$f" "status/runs/$n/"
    done
  done
  { date -u +"%F %T UTC"; echo "job: ${cur_job:-none}";
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null;
    df -h /workspace | tail -1; } > status/heartbeat.txt
}

push() {
  sync_logs
  git add -A status jobs >/dev/null 2>&1
  git commit -qm "$1" >/dev/null 2>&1 || true
  for i in 1 2 3; do
    git pull --rebase -q >/dev/null 2>&1 && git push -q >/dev/null 2>&1 && break
    sleep 5
  done
  last_push=$(date +%s)
}

echo "agent started $(date -u)"
push "agent started"
while true; do
  git pull --rebase -q >/dev/null 2>&1 || true
  # cancel request
  if [ -n "$cur_job" ] && [ -f jobs/cancel ] && grep -qx "$cur_job" jobs/cancel; then
    kill -- -"$cur_pid" 2>/dev/null || kill "$cur_pid" 2>/dev/null
    echo "=== cancelled $(date -u)" >> "status/$cur_job.log"
    : > jobs/cancel
  fi
  # finished?
  if [ -n "$cur_pid" ] && ! kill -0 "$cur_pid" 2>/dev/null; then
    wait "$cur_pid"; rc=$?
    echo "=== exit code $rc  $(date -u)" >> "status/$cur_job.log"
    git mv -f "jobs/queue/$cur_job.sh" "jobs/done/$cur_job.sh" >/dev/null 2>&1 \
      || mv -f "jobs/queue/$cur_job.sh" "jobs/done/" 2>/dev/null
    push "done $cur_job (exit $rc)"
    cur_pid=""; cur_job=""
  fi
  # start the next job
  if [ -z "$cur_pid" ]; then
    next=$(ls jobs/queue/*.sh 2>/dev/null | sort | head -1)
    if [ -n "$next" ]; then
      cur_job=$(basename "$next" .sh)
      echo "=== start $(date -u)" > "status/$cur_job.log"
      ( cd code && exec setsid bash "$REPO/$next" ) >> "status/$cur_job.log" 2>&1 &
      cur_pid=$!
      push "start $cur_job"
    fi
  fi
  [ $(( $(date +%s) - last_push )) -ge "$PUSH_EVERY" ] && push "progress ${cur_job:-idle}"
  sleep 60
done
