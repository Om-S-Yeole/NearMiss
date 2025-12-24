# FastAPI Service Example Usage

This notebook demonstrates how to use the NearMiss FastAPI service for satellite collision prediction.

## Prerequisites

Make sure the FastAPI server is running:

```bash
cd src
python start_api_server.py
```

## Installation

```python
# Install requests library if not already installed
# pip install requests
```

## Import Libraries

```python
import requests
import json
from datetime import datetime
```

## Example 1: Basic Collision Prediction

```python
# API endpoint
url = "http://localhost:8000/predict"

# Sample TLE data (International Space Station and a nearby satellite)
payload = {
    "satellite1": {
        "tle_line1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "tle_line2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    },
    "satellite2": {
        "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
        "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500"
    },
    "time_window_hours": 24.0
}

# Make the request
response = requests.post(url, json=payload)

# Check if successful
if response.status_code == 200:
    result = response.json()
    
    print("=" * 60)
    print("COLLISION PREDICTION RESULTS")
    print("=" * 60)
    print(f"Collision Probability: {result['collision_probability']:.6f}")
    print(f"Minimum Distance: {result['minimum_distance_km']:.2f} km")
    print(f"Filter Rejected: {result['filter_rejected']}")
    print(f"Analysis Period: {result['analysis_start_time']}")
    print(f"                 to {result['analysis_end_time']}")
    print(f"\nMessage: {result['message']}")
    print("=" * 60)
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## Example 2: With Custom Satellite Radii

```python
# Specify exact satellite radii
payload_with_radii = {
    "satellite1": {
        "tle_line1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "tle_line2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    },
    "satellite2": {
        "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
        "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500"
    },
    "time_window_hours": 48.0,
    "satellite1_radius": 7.5,  # meters
    "satellite2_radius": 5.0,  # meters
    "distance_threshold": 15.0  # km
}

response = requests.post(url, json=payload_with_radii)
result = response.json()

print(f"Collision Probability: {result['collision_probability']:.6f}")
print(f"Minimum Distance: {result['minimum_distance_km']:.2f} km")
```

## Example 3: Check API Health

```python
# Check if API is running and healthy
health_url = "http://localhost:8000/health"
health_response = requests.get(health_url)

health = health_response.json()
print(f"API Status: {health['status']}")
print(f"CUDA Available: {health['cuda_available']}")
print(f"Device: {health['device']}")
```

## Example 4: Batch Predictions

```python
# Multiple satellite pairs
satellite_pairs = [
    {
        "satellite1": {
            "tle_line1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            "tle_line2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        },
        "satellite2": {
            "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
            "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500"
        },
        "time_window_hours": 24.0
    },
    # Add more pairs as needed
]

results = []
for i, pair in enumerate(satellite_pairs):
    print(f"Processing pair {i+1}...")
    response = requests.post(url, json=pair)
    if response.status_code == 200:
        results.append(response.json())
    else:
        print(f"Error processing pair {i+1}: {response.status_code}")

# Analyze results
high_risk_pairs = [r for r in results if r['collision_probability'] > 0.1]
print(f"\nFound {len(high_risk_pairs)} high-risk pairs")
```

## Example 5: Error Handling

```python
# Example with invalid TLE data
invalid_payload = {
    "satellite1": {
        "tle_line1": "INVALID",  # Too short
        "tle_line2": "INVALID"
    },
    "satellite2": {
        "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
        "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500"
    }
}

try:
    response = requests.post(url, json=invalid_payload)
    if response.status_code != 200:
        error = response.json()
        print(f"Error {response.status_code}: {error['detail']}")
except Exception as e:
    print(f"Exception: {str(e)}")
```

## Visualizing Results (Optional)

```python
# If you have matplotlib installed
import matplotlib.pyplot as plt

# Collect predictions from multiple time windows
time_windows = [6, 12, 24, 48, 72, 96, 120, 144, 168]
probabilities = []

for hours in time_windows:
    payload['time_window_hours'] = hours
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        probabilities.append(result['collision_probability'])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_windows, probabilities, marker='o')
plt.xlabel('Time Window (hours)')
plt.ylabel('Collision Probability')
plt.title('Collision Probability vs Time Window')
plt.grid(True)
plt.show()
```

## Notes

- Always ensure the FastAPI server is running before making requests
- TLE lines must be valid and at least 69 characters long
- Maximum time window is 168 hours (7 days)
- For production use, consider implementing retry logic and rate limiting
- The API returns collision probability between 0.0 (no risk) and 1.0 (certain collision)
