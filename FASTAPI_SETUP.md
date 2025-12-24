# NearMiss FastAPI Service - Setup Summary

## What Was Created

This FastAPI service enables satellite collision prediction via REST API endpoints. The following files were created/modified:

### Core Files

1. **src/app/app.py** - Main FastAPI application
   - `/predict` POST endpoint for collision prediction
   - `/health` GET endpoint for health checks
   - `/` GET endpoint for API information
   - Validates TLE data and calculates collision probability and minimum distance

2. **src/start_api_server.py** - Server startup script
   - Easy way to start the server with command-line options
   - Supports development and production modes

3. **tests/test_api.py** - API test suite
   - Tests all endpoints
   - Validates request/response format
   - Demonstrates API usage

### Documentation

4. **docs/API_USAGE.md** - Complete API documentation
   - Detailed endpoint descriptions
   - Request/response schemas
   - Usage examples in multiple languages (Python, cURL, JavaScript)

5. **docs/FASTAPI_QUICKSTART.md** - Quick start guide
   - Installation instructions
   - Multiple ways to run the server
   - Troubleshooting tips

### Dependencies

6. **requirements.txt** - Updated with FastAPI dependencies
   - Added: fastapi==0.115.0
   - Added: uvicorn==0.34.0

## How to Use

### Step 1: Install Dependencies

```bash
pip install fastapi uvicorn
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Start the Server

From the project root:

```bash
cd src
python start_api_server.py
```

Or with auto-reload for development:

```bash
python start_api_server.py --reload
```

### Step 3: Access the API

- **Interactive Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health

### Step 4: Make Predictions

**Python Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
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
)

result = response.json()
print(f"Collision Probability: {result['collision_probability']}")
print(f"Minimum Distance: {result['minimum_distance_km']} km")
print(f"Message: {result['message']}")
```

## API Endpoint Details

### POST /predict

**Request Body:**
```json
{
  "satellite1": {
    "tle_line1": "string (69+ chars)",
    "tle_line2": "string (69+ chars)"
  },
  "satellite2": {
    "tle_line1": "string (69+ chars)",
    "tle_line2": "string (69+ chars)"
  },
  "time_window_hours": 24.0,
  "satellite1_radius": 5.0,
  "satellite2_radius": 5.0,
  "distance_threshold": 10.0
}
```

**Response:**
```json
{
  "collision_probability": 0.023,
  "minimum_distance_km": 5.34,
  "filter_rejected": false,
  "analysis_start_time": "2025-12-24T10:30:00",
  "analysis_end_time": "2025-12-25T10:30:00",
  "message": "Low collision risk. Minimum distance: 5.34 km"
}
```

## Key Features

✅ **SGP4 Propagation**: Uses industry-standard SGP4 algorithm for orbital predictions
✅ **Physical Algorithm**: Implements close approach detection with collision probability calculation
✅ **Input Validation**: Validates TLE format and parameter constraints
✅ **Error Handling**: Returns appropriate HTTP status codes and error messages
✅ **Interactive Docs**: Auto-generated Swagger UI documentation
✅ **Flexible Configuration**: Optional parameters for satellite radii and time windows

## Testing

Run the test suite:

```bash
# Make sure the server is running first
cd tests
python test_api.py
```

## Production Deployment

For production use with multiple workers:

```bash
python start_api_server.py --workers 4
```

Or directly with uvicorn:

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Notes

- The API uses the existing `close_approach_physical_algorithm_sgp4` function from the nearmiss package
- TLE lines must be at least 69 characters (standard TLE format)
- Maximum time window is 168 hours (7 days)
- If satellite radii are not provided, random values between 1-10 meters are used
- The algorithm includes filters to quickly reject low-risk satellite pairs

## Next Steps

1. **Install dependencies**: `pip install fastapi uvicorn`
2. **Start the server**: `python src/start_api_server.py`
3. **Visit the docs**: http://localhost:8000/docs
4. **Try it out**: Use the interactive API documentation to test predictions

For more information, see:
- Full API documentation: docs/API_USAGE.md
- Quick start guide: docs/FASTAPI_QUICKSTART.md
