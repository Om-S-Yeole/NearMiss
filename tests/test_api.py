"""
Test script for the NearMiss FastAPI collision prediction service.

This script demonstrates how to use the API endpoint and verifies its functionality.
"""

import json

import requests


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)

    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_root_endpoint(base_url: str = "http://localhost:8000"):
    """Test the root endpoint."""
    print("=" * 60)
    print("Testing Root Endpoint")
    print("=" * 60)

    response = requests.get(f"{base_url}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_collision_prediction(base_url: str = "http://localhost:8000"):
    """Test the collision prediction endpoint with sample TLE data."""
    print("=" * 60)
    print("Testing Collision Prediction Endpoint")
    print("=" * 60)

    # Sample TLE data (ISS and similar orbit satellite)
    data = {
        "satellite1": {
            "tle_line1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            "tle_line2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        },
        "satellite2": {
            "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
            "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500",
        },
        "time_window_hours": 24.0,
        "satellite1_radius": 5.0,
        "satellite2_radius": 5.0,
        "distance_threshold": 10.0,
    }

    print(f"Request Payload:")
    print(json.dumps(data, indent=2))
    print()

    try:
        response = requests.post(f"{base_url}/predict", json=data)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response:")
            print(json.dumps(result, indent=2))
            print()
            print("=" * 60)
            print("PREDICTION RESULTS:")
            print("=" * 60)
            print(f"Collision Probability: {result['collision_probability']:.6f}")
            print(f"Minimum Distance: {result['minimum_distance_km']:.2f} km")
            print(f"Filter Rejected: {result['filter_rejected']}")
            print(
                f"Analysis Period: {result['analysis_start_time']} to {result['analysis_end_time']}"
            )
            print(f"Message: {result['message']}")
        else:
            print(f"Error Response:")
            print(json.dumps(response.json(), indent=2))

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server.")
        print("Make sure the FastAPI server is running at", base_url)
    except Exception as e:
        print(f"ERROR: {str(e)}")

    print()


def test_invalid_tle(base_url: str = "http://localhost:8000"):
    """Test the API with invalid TLE data to verify error handling."""
    print("=" * 60)
    print("Testing Error Handling (Invalid TLE)")
    print("=" * 60)

    data = {
        "satellite1": {
            "tle_line1": "INVALID TLE",  # Too short
            "tle_line2": "ALSO INVALID",
        },
        "satellite2": {
            "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
            "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500",
        },
        "time_window_hours": 24.0,
    }

    try:
        response = requests.post(f"{base_url}/predict", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"ERROR: {str(e)}")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  NearMiss API Test Suite".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    base_url = "http://localhost:8000"

    # Run tests
    try:
        test_root_endpoint(base_url)
        test_health_check(base_url)
        test_collision_prediction(base_url)
        test_invalid_tle(base_url)

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")


if __name__ == "__main__":
    main()
