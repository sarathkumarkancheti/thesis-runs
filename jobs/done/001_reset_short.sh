# Short run of the supervisor's reset shooting (epochs x0.1, anchor 5 epochs):
# measures time per epoch and the window rejection rate before a full run.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, torchdiffeq; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python reset_shooting.py --data_dir "$DATA_DIR" --out_dir "$RUNS/reset_short" --epochs_scale 0.1 --print_every 1
