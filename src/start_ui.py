"""
Script to start the NearMiss Streamlit UI.

Usage:
    python start_ui.py [--port PORT]
"""

import argparse
import sys
import os


def main():
    """Start the Streamlit UI."""
    parser = argparse.ArgumentParser(description="Start the NearMiss Streamlit UI")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the UI to (default: localhost)",
    )

    args = parser.parse_args()

    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)

    # Get the path to the streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "ui", "streamlit_app.py")

    if not os.path.exists(app_path):
        print(f"Error: Could not find streamlit_app.py at {app_path}")
        sys.exit(1)

    print("=" * 60)
    print("Starting NearMiss Streamlit UI")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print("=" * 60)
    print()
    print(f"UI will be available at: http://{args.host}:{args.port}")
    print()
    print("Press CTRL+C to stop the UI")
    print("=" * 60)

    # Start Streamlit
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        f"--server.port={args.port}",
        f"--server.address={args.host}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
