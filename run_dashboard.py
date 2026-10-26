"""Launcher for the Industrial Safety Intelligence Streamlit Console."""

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int = 8501, max_attempts: int = 10) -> int:
    port = start_port
    while port < start_port + max_attempts:
        if not is_port_in_use(port):
            return port
        port += 1
    return start_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Safety Console")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8501")), help="Port for dashboard")
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parent / "industrial_safety_system" / "ui" / "dashboard.py"
    target_port = args.port

    if is_port_in_use(target_port):
        alt_port = find_available_port(target_port + 1)
        print(f"[WARN] Port {target_port} is currently in use.")
        print(f"[INFO] Automatically switching dashboard to available port: {alt_port}")
        target_port = alt_port

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(target_port),
        "--server.headless",
        "false",
        "--theme.base",
        "dark",
    ]
    print("=" * 70)
    print(f"[START] Launching Industrial Safety Dashboard on http://localhost:{target_port}...")
    print("=" * 70)
    subprocess.run(cmd)
