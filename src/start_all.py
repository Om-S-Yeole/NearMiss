"""
Script to start both the FastAPI backend and Streamlit UI.

This script will start:
1. FastAPI server on port 8000
2. Streamlit UI on port 8501

Usage:
    python start_all.py
"""

import subprocess
import sys
import time
import signal
import os


processes = []


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shutdown both services."""
    print("\n" + "=" * 60)
    print("Shutting down services...")
    print("=" * 60)
    for proc in processes:
        proc.terminate()
    sys.exit(0)


def main():
    """Start both FastAPI and Streamlit services."""
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 60)
    print("Starting NearMiss Full Application")
    print("=" * 60)
    print()
    print("This will start:")
    print("1. FastAPI Backend on http://localhost:8000")
    print("2. Streamlit UI on http://localhost:8501")
    print()
    print("Press CTRL+C to stop all services")
    print("=" * 60)
    print()

    try:
        # Start FastAPI backend
        print("🚀 Starting FastAPI backend...")
        api_process = subprocess.Popen(
            [sys.executable, "start_api_server.py"],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(api_process)
        print("✅ FastAPI backend started (PID: {})".format(api_process.pid))

        # Wait a bit for API to start
        time.sleep(3)

        # Start Streamlit UI
        print("🎨 Starting Streamlit UI...")
        ui_process = subprocess.Popen(
            [sys.executable, "start_ui.py"],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(ui_process)
        print("✅ Streamlit UI started (PID: {})".format(ui_process.pid))

        print()
        print("=" * 60)
        print("✅ All services are running!")
        print("=" * 60)
        print()
        print("📍 Access points:")
        print("   • Streamlit UI: http://localhost:8501")
        print("   • FastAPI Backend: http://localhost:8000")
        print("   • API Docs: http://localhost:8000/docs")
        print()
        print("Press CTRL+C to stop all services")
        print("=" * 60)

        # Keep the script running
        while True:
            time.sleep(1)
            # Check if any process has died
            for proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️ Process {proc.pid} has stopped unexpectedly")
                    signal_handler(None, None)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        for proc in processes:
            proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
