#!/usr/bin/env python3
"""
Upload mc_maze.nwb file using local HTTP server + wget on remote
INSTRUCTIONS:
1. Run this script - it will start an HTTP server
2. The script will automatically connect to Fly and download the file
3. Press Ctrl+C when done
"""
import http.server
import socketserver
import subprocess
import threading
import time
import os
import socket

# Configuration
DATA_FILE = r"c:\Users\guzzi\Desktop\Projects\DEV-ACTIF\NeuraLink\PhantomX\data\mc_maze.nwb"
PORT = 8765
APP_NAME = "phantomx-gpu"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(DATA_FILE), **kwargs)
    
    def log_message(self, format, *args):
        print(f"[HTTP] {format % args}")

def get_local_ip():
    """Get local IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def start_server():
    """Start HTTP server in background"""
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🌐 HTTP server started on port {PORT}")
        httpd.serve_forever()

def main():
    print("=" * 70)
    print("  PhantomX Data Upload via HTTP")
    print("=" * 70)
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file not found: {DATA_FILE}")
        return
    
    file_size = os.path.getsize(DATA_FILE) / 1024 / 1024
    file_name = os.path.basename(DATA_FILE)
    local_ip = get_local_ip()
    
    print(f"\n📦 File: {file_name} ({file_size:.2f} MB)")
    print(f"🖥️  Local IP: {local_ip}")
    print(f"🔌 Port: {PORT}")
    
    # Start HTTP server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    
    print(f"\n🚀 Starting Fly proxy...")
    # Start flyctl proxy in background
    proxy_process = subprocess.Popen(
        ['flyctl', 'proxy', f'{PORT}:{PORT}', '--app', APP_NAME],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("   Waiting for proxy to be ready...")
    time.sleep(5)
    
    # Now download on remote machine via localhost
    print(f"\n📥 Downloading file on remote machine...")
    download_url = f"http://host.docker.internal:{PORT}/{file_name}"
    
    # Try with curl first
    download_cmd = f'su - phantomx -c "cd ~/data && curl -f -o mc_maze.nwb {download_url} || wget -O mc_maze.nwb {download_url}"'
    
    result = subprocess.run(
        ['flyctl', 'ssh', 'console', '--app', APP_NAME, '-C', download_cmd],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✅ Download successful!")
    else:
        print("\n⚠️  Download may have failed, checking...")
    
    # Verify
    print(f"\n📊 Verifying upload...")
    verify_result = subprocess.run(
        ['flyctl', 'ssh', 'console', '--app', APP_NAME, '-C',
         'su - phantomx -c "ls -lh ~/data/mc_maze.nwb"'],
        capture_output=True,
        text=True
    )
    print(verify_result.stdout)
    
    # Cleanup
    proxy_process.terminate()
    print("\n✨ Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
