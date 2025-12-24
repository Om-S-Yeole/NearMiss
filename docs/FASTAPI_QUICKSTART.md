# FastAPI Service Quick Start Guide

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

Or install just the API dependencies:

```bash
pip install fastapi uvicorn pydantic
```

## Running the Server

### Option 1: Using the startup script (Recommended)

```bash
cd src
python start_api_server.py
```

With custom options:

```bash
# Development mode with auto-reload
python start_api_server.py --reload

# Custom host and port
python start_api_server.py --host 127.0.0.1 --port 8080

# Production mode with multiple workers
python start_api_server.py --workers 4
```

### Option 2: Direct uvicorn command

```bash
cd src
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

### Option 3: Run as Python module

```bash
cd src/app
python app.py
```

## Accessing the API

Once the server is running:

- **API Root**: http://localhost:8000/
- **Interactive Documentation (Swagger)**: http://localhost:8000/docs
- **Alternative Documentation (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Making Predictions

### Example Request (Python)

```python
import requests

url = "http://localhost:8000/predict"

data = {
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

response = requests.post(url, json=data)
result = response.json()

print(f"Collision Probability: {result['collision_probability']}")
print(f"Minimum Distance: {result['minimum_distance_km']} km")
```

### Example Request (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "satellite1": {
      "tle_line1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
      "tle_line2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    },
    "satellite2": {
      "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
      "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500"
    },
    "time_window_hours": 24.0
  }'
```

## Testing the API

Run the test script:

```bash
cd tests
python test_api.py
```

Make sure the server is running before executing tests.

## Request Parameters

### Required:
- `satellite1.tle_line1`: First TLE line for satellite 1
- `satellite1.tle_line2`: Second TLE line for satellite 1
- `satellite2.tle_line1`: First TLE line for satellite 2
- `satellite2.tle_line2`: Second TLE line for satellite 2

### Optional:
- `time_window_hours`: Analysis time window (default: 24 hours, max: 168 hours)
- `satellite1_radius`: Radius in meters (default: random 1-10m)
- `satellite2_radius`: Radius in meters (default: random 1-10m)
- `distance_threshold`: Minimum distance threshold in km (default: 10 km)

## Response Fields

- `collision_probability`: Probability of collision (0.0 to 1.0)
- `minimum_distance_km`: Minimum distance between satellites in kilometers
- `filter_rejected`: Whether the pair was rejected by filters (low risk if true)
- `analysis_start_time`: Start of analysis window
- `analysis_end_time`: End of analysis window
- `message`: Human-readable risk assessment

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, specify a different port:

```bash
python start_api_server.py --port 8080
```

### Import Errors

Make sure you're running from the correct directory:

```bash
cd src
python start_api_server.py
```

### Module Not Found

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Add the project root to PYTHONPATH if needed:

```bash
# Windows PowerShell
$env:PYTHONPATH = "D:\Documents\NearMiss\NearMiss\src"

# Linux/Mac
export PYTHONPATH=/path/to/NearMiss/src
```

## For More Information

See the detailed API documentation: [docs/API_USAGE.md](../docs/API_USAGE.md)
