# Fly.io GPU Deployment Issues - PhantomX

**Date:** January 19, 2026  
**App Name:** phantomx (previously phantomx-gpu)  
**Target:** A100-40GB GPU machine in ORD region

---

## Problem Summary

Multiple attempts to deploy PhantomX to Fly.io GPU infrastructure failed at various stages. The application never successfully started despite successful Docker image builds.

---

## Deployment Attempts Timeline

### Attempt 1: Initial phantomx-gpu app
- **Status:** Deployment succeeded, but machine auto-stopped
- **Configuration:** Using `fly.gpu.toml` with `Dockerfile.gpu`
- **Issues:**
  - Machine would start but immediately stop due to `min_machines_running = 0` and `auto_stop_machines = "stop"`
  - Data file (mc_maze.nwb) needed to be uploaded separately
  - Successful upload of 28MB data file via `flyctl ssh sftp put`
  - Updated exp11 script path from Windows path to Linux path: `/home/phantomx/data/mc_maze.nwb`

### Attempt 2: Fresh deployment after app destruction
- **Status:** FAILED - App configuration overwritten during `flyctl launch`
- **Root Cause:** 
  ```bash
  flyctl launch --config fly.gpu.toml --no-deploy
  ```
  This command **OVERWROTE** the carefully crafted `fly.gpu.toml` with default FastAPI configuration:
  - Replaced GPU config with standard 1GB RAM machine
  - Changed dockerfile reference from `Dockerfile.gpu` to default
  - Removed all GPU-specific settings (A100-40GB, swap, mounts)

### Attempt 3: After restoring fly.gpu.toml
- **Status:** FAILED - Smoke checks failed, machine stopped
- **Build:** ✅ Successful (192 MB image)
- **Deployment:** ❌ Failed
- **Error Messages:**
  ```
  Warn: got an error retrieving the logs so we can't show you what failed
  ✖ Failed: smoke checks for d89d1e6f6d2e98 failed: failed to get VM
  ```
- **Machine ID:** d89d1e6f6d2e98
- **State:** Stopped immediately after creation

---

## Configuration Files

### fly.gpu.toml (Correct GPU Configuration)
```toml
app = "phantomx"
primary_region = "ord"  # Chicago - has GPUs available

[build]
  dockerfile = "Dockerfile.gpu"
  [build.args]
    NONROOT_USER = "phantomx"

[[vm]]
  size = "a100-40gb"

swap_size_mb = 32768  # 32GB swap

[mounts]
source = "phantomx_data"
destination = "/home/phantomx"

[http_service]
  internal_port = 8888
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
```

### Dockerfile.gpu
```dockerfile
FROM ubuntu:22.04

RUN apt update -q && apt install -y \
    python3 python3-pip python3-venv python3-wheel \
    git nano wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash phantomx

COPY --chmod=0755 ./gpu-entrypoint.sh ./entrypoint.sh
COPY --chmod=0755 ./gpu-post-init.sh ./post-init.sh
COPY --chown=phantomx:phantomx ./python /app/python
COPY --chown=phantomx:phantomx ./requirements.txt /app/requirements.txt

CMD ["bash", "./entrypoint.sh", "phantomx"]
```

---

## Root Causes Identified

### 1. **Entrypoint Script Issues**
The container likely fails health checks because:
- `gpu-entrypoint.sh` expects to initialize GPU with `nvidia-smi`
- `gpu-post-init.sh` ends with `sleep inf` - may not be reached or working correctly
- No proper health check endpoint configured for Fly.io to verify

### 2. **Health Check / Smoke Test Failures**
Fly.io's smoke checks expect:
- HTTP service to respond on configured port (8888)
- But our app doesn't expose HTTP - it's SSH-only
- Mismatch between declared service and actual container behavior

### 3. **Volume Mount Timing**
- Volume `phantomx_data` created during deployment
- Mount at `/home/phantomx` may not be ready when entrypoint runs
- Could cause initialization failures

### 4. **Missing Logs**
Critical issue: Cannot retrieve logs to debug failures
```
Warn: got an error retrieving the logs so we can't show you what failed
```
This makes debugging nearly impossible.

---

## What Worked

✅ Docker image builds successfully (192 MB)  
✅ Data file upload via `flyctl ssh sftp put` (29,207,528 bytes)  
✅ Volume creation (100 GB `phantomx_data`)  
✅ Python code path updates for Linux environment  

---

## What Failed

❌ Machine startup and health checks  
❌ Log retrieval for debugging  
❌ HTTP service configuration mismatch  
❌ Entrypoint script execution/completion  
❌ `flyctl launch` command corrupts custom configuration  

---

## Potential Solutions (Not Implemented)

### Option 1: Remove HTTP Service Requirement
```toml
# Comment out or remove [http_service] section entirely
# Fly may require this for deployments though
```

### Option 2: Add Minimal HTTP Health Endpoint
Add to `gpu-post-init.sh`:
```bash
# Start a simple health check server
python3 -m http.server 8888 &
# Then continue with sleep inf
sleep inf
```

### Option 3: Fix Entrypoint to Background Post-Init
```bash
# In gpu-entrypoint.sh
su -c "bash ./post-init.sh &" $USERNAME
# Don't wait for it to complete
```

### Option 4: Use Different Deployment Strategy
- Use `flyctl machine run` instead of `flyctl deploy`
- Manually create machine with exact specifications
- Skip health checks with `--skip-health-checks` flag

### Option 5: Start Machine Manually
```bash
flyctl machine start d89d1e6f6d2e98 --app phantomx
flyctl ssh console --app phantomx
```

---

## Lessons Learned

1. **Never use `flyctl launch` with existing config** - It overwrites everything
2. **GPU deployments need special handling** - Standard health checks don't work
3. **SSH-only apps conflict with Fly.io expectations** - HTTP service seems mandatory
4. **Log access is critical** - Without logs, debugging is impossible
5. **Entrypoint scripts must complete quickly** - Long-running init breaks deployments

---

## Current State

- **App:** phantomx
- **Machine:** d89d1e6f6d2e98 (stopped)
- **Image:** registry.fly.io/phantomx:deployment-01KFCKZA1T5A9Z1BMGP463MWTJ
- **Volume:** phantomx_data (100 GB, mounted at /home/phantomx)
- **Data:** mc_maze.nwb uploaded and ready
- **Code:** exp11_close_gap.py updated with correct Linux paths

**Status:** Blocked - Cannot run experiments until deployment issues resolved

---

## Recommended Next Steps

1. Try starting machine manually and SSH in to debug
2. Check if GPU is actually allocated to the stopped machine
3. Add simple HTTP health endpoint
4. Simplify entrypoint script to minimize startup time
5. Consider alternative GPU cloud providers (Lambda Labs, RunPod, Vast.ai)
6. OR: Run experiments locally if GPU available

---

## Alternative: Local Execution

If Fly.io continues to be problematic, run exp11 locally:
```bash
# Assuming local GPU available
cd c:\Users\guzzi\Desktop\Projects\DEV-ACTIF\NeuraLink\PhantomX\python
python exp11_close_gap.py 2>&1 | tee exp11_output.log
```

This requires:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- All dependencies from requirements.txt
- ~28 MB data file already present
