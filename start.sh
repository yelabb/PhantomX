#!/bin/bash
# PhantomX GPU - Simple Startup Script
# Starts health check server and keeps container alive for SSH access

echo "=== PhantomX GPU Starting ==="

# Quick GPU check (non-blocking)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU not available yet"

# Start minimal health check server (Fly.io needs this)
python3 -c "
from flask import Flask
app = Flask(__name__)

@app.route('/')
def health():
    return 'PhantomX GPU Ready'

@app.route('/health')
def health_check():
    return 'OK', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
" &

echo ""
echo "=== Health server started on port 8888 ==="
echo "=== SSH in with: fly ssh console -a phantomx ==="
echo "=== Then run: cd /home/phantomx/project/python && python exp11_close_gap.py ==="
echo ""

# Keep container alive
sleep infinity
