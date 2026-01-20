# Fly.io GPU Deployment Guide

Quick reference for training PhantomX experiments on Fly.io GPU infrastructure.

## Quick Start

```bash
# SSH into the GPU machine
flyctl ssh console --app phantomx

# Run an experiment
cd /home/phantomx/project/python
python3 exp11_close_gap.py
```

## Common Commands

### Machine Management

```bash
# Check machine status
flyctl machines list --app phantomx

# Start machine (if stopped/suspended)
flyctl machines start <MACHINE_ID> --app phantomx

# Stop machine (to save costs)
flyctl machines stop <MACHINE_ID> --app phantomx

# SSH into machine
flyctl ssh console --app phantomx

# Run a single command via SSH
flyctl ssh console --app phantomx --command "nvidia-smi"
```

### Update Code (After GitHub Push)

```bash
# SSH in and pull latest changes
flyctl ssh console --app phantomx --command "cd /home/phantomx/project && git pull"
```

### Check GPU Status

```bash
flyctl ssh console --app phantomx --command "nvidia-smi"
flyctl ssh console --app phantomx --command "python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
```

### Run Experiments

```bash
# Interactive session (recommended for long runs)
flyctl ssh console --app phantomx
cd /home/phantomx/project/python
python3 exp11_close_gap.py

# Or run directly (may timeout for long experiments)
flyctl ssh console --app phantomx --command "cd /home/phantomx/project/python && python3 exp11_close_gap.py"
```

### Background Training (Persistent)

For long experiments that should survive SSH disconnection:

```bash
flyctl ssh console --app phantomx

# Use nohup or screen
cd /home/phantomx/project/python
nohup python3 exp11_close_gap.py > exp11.log 2>&1 &

# Check progress
tail -f exp11.log
```

## Costs & Auto-Scaling

- **A100-40GB**: ~$2.50/hour when running
- **Auto-suspend**: Machine suspends when idle (no SSH activity)
- **Auto-start**: Machine wakes when you SSH in

To minimize costs:
1. Stop machine when done: `flyctl machines stop <ID> --app phantomx`
2. Or let it auto-suspend after idle timeout

## Troubleshooting

### Machine Won't Start
```bash
# Check logs
flyctl logs --app phantomx

# Try restarting
flyctl machines restart <MACHINE_ID> --app phantomx
```

### Out of Memory
```bash
# Check GPU memory
flyctl ssh console --app phantomx --command "nvidia-smi"

# Reduce batch size in experiment config
```

### Code Not Updated
```bash
# Pull latest from GitHub
flyctl ssh console --app phantomx --command "cd /home/phantomx/project && git fetch && git reset --hard origin/main"
```

## Redeploy (Full Rebuild)

If you need to rebuild the Docker image (e.g., new dependencies):

```bash
cd PhantomX
flyctl deploy --config fly.gpu.toml --remote-only
```

**Note**: This rebuilds the image and clones fresh from GitHub.
