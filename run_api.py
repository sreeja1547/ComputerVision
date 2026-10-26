"""Launcher for the Industrial Safety Intelligence FastAPI Service."""

import argparse
import os
import socket
import sys
import uvicorn
from pathlib import Path

# Add workspace directory to python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    port = start_port
    while port < start_port + max_attempts:
        if not is_port_in_use(port):
            return port
        port += 1
    return start_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Safety REST API")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to run API on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()

    target_port = args.port
    if is_port_in_use(target_port):
        alt_port = find_available_port(target_port + 1)
        print(f"[WARN] Port {target_port} is already in use by another process.")
        print(f"[INFO] Automatically switching to available port: {alt_port}")
        target_port = alt_port

    print("=" * 70)
    print(f"[START] Starting Industrial Inspection REST API on http://localhost:{target_port}")
    print(f"[DOCS]  Interactive Swagger Docs: http://localhost:{target_port}/docs")
    print("=" * 70)

    uvicorn.run("industrial_safety_system.api.app:app", host=args.host, port=target_port, reload=False)
