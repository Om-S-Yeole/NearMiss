"""
Script to start the NearMiss FastAPI server.

Usage:
    python start_api_server.py [--host HOST] [--port PORT] [--reload]
"""

import argparse
import sys


def main():
    """Start the FastAPI server with uvicorn."""
    parser = argparse.ArgumentParser(
        description="Start the NearMiss FastAPI collision prediction server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, not compatible with --reload)",
    )

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed.")
        print("Please install it with: pip install uvicorn")
        sys.exit(1)

    print("=" * 60)
    print("Starting NearMiss API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    if not args.reload:
        print(f"Workers: {args.workers}")
    print("=" * 60)
    print()
    print(
        f"API Documentation: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs"
    )
    print(
        f"API Root: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/"
    )
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)

    # Start the server
    uvicorn.run(
        "app.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
