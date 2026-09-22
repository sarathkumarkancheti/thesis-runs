# thesis-runs
Remote job relay for the RunPod GPU. `agent.sh` runs on the pod: it pulls
`jobs/queue/*.sh`, runs them one at a time (working directory `code/`), and
pushes logs to `status/`. Checkpoints stay on the pod in `/workspace/runs`.
