srun -p gpu \
    --job-name=ano_train \
    --nodes=1 \
    --gpus=1 \
    --cpus-per-task=10 \
    --time=10:00:00 \
    --kill-on-bad-exit=1 \
    --mem-per-gpu=32G \
    python -u models/riad/riad.py