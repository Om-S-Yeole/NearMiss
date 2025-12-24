# NearMiss FastAPI Service

## Overview

The NearMiss FastAPI service provides a REST API for predicting satellite collision risks using Two-Line Element (TLE) data. The service calculates collision probability and minimum distance between satellite pairs using SGP4 propagation and physical algorithms.

## Installation

Ensure you have all required dependencies installed:

```bash
pip install fastapi uvicorn pydantic
```

## Running the Server

### Development Mode

```bash
# From the project root directory
cd src/app
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Returns basic API information.

**Response:**
```json
{
  "name": "NearMiss Satellite Collision Prediction API",
  "version": "1.0.0",
  "description": "Predict collision probability between satellites using TLE data",
  "endpoints": {
    "/predict": "POST - Predict collision between two satellites",
    "/health": "GET - Check API health status"
  }
}
```

### 2. Health Check
**GET** `/health`

Check if the API is running and GPU availability.

**Response:**
```json
{
  "status": "healthy",
  "cuda_available": false,
  "device": "cpu"
}
```

### 3. Collision Prediction
**POST** `/predict`

Predict collision probability and minimum distance between two satellites.

**Request Body:**
```json
{
  "satellite1": {
    "tle_line1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "tle_line2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
  },
  "satellite2": {
    "tle_line1": "1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234",
    "tle_line2": "2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500"
  },
  "time_window_hours": 24.0,
  "satellite1_radius": 5.0,
  "satellite2_radius": 5.0,
  "distance_threshold": 10.0
}
```

**Request Parameters:**
- `satellite1` (required): TLE data for the first satellite
  - `tle_line1`: First line of TLE (69+ characters)
  - `tle_line2`: Second line of TLE (69+ characters)
- `satellite2` (required): TLE data for the second satellite
  - `tle_line1`: First line of TLE
  - `tle_line2`: Second line of TLE
- `time_window_hours` (optional): Analysis time window in hours (default: 24.0, max: 168)
- `satellite1_radius` (optional): Radius of satellite 1 in meters (default: random 1-10m)
- `satellite2_radius` (optional): Radius of satellite 2 in meters (default: random 1-10m)
- `distance_threshold` (optional): Minimum distance threshold in km (default: 10.0)

**Response:**
```json
{
  "collision_probability": 0.0234,
  "minimum_distance_km": 5.342,
  "filter_rejected": false,
  "analysis_start_time": "2025-12-24T10:30:00",
  "analysis_end_time": "2025-12-25T10:30:00",
  "message": "Low collision risk. Minimum distance: 5.34 km"
}
```

**Response Fields:**
- `collision_probability`: Probability of collision (0.0 to 1.0)
- `minimum_distance_km`: Minimum distance between satellites in kilometers
- `filter_rejected`: Whether the pair was rejected by initial filters (if true, collision risk is very low)
- `analysis_start_time`: Start of the analysis time window (ISO format)
- `analysis_end_time`: End of the analysis time window (ISO format)
- `message`: Human-readable message about collision risk

## Usage Examples

### Python (using requests)

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
    "time_window_hours": 24.0,
    "satellite1_radius": 5.0,
    "satellite2_radius": 5.0,
    "distance_threshold": 10.0
}

response = requests.post(url, json=data)
result = response.json()

print(f"Collision Probability: {result['collision_probability']}")
print(f"Minimum Distance: {result['minimum_distance_km']} km")
print(f"Message: {result['message']}")
```

### cURL

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

### JavaScript (fetch)

```javascript
const url = 'http://localhost:8000/predict';

const data = {
  satellite1: {
    tle_line1: '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927',
    tle_line2: '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537'
  },
  satellite2: {
    tle_line1: '1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234',
    tle_line2: '2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500'
  },
  time_window_hours: 24.0
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(result => {
    console.log('Collision Probability:', result.collision_probability);
    console.log('Minimum Distance:', result.minimum_distance_km, 'km');
    console.log('Message:', result.message);
  });
```

## API Documentation

Once the server is running, you can access:

- **Interactive API documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative API documentation (ReDoc)**: `http://localhost:8000/redoc`

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Invalid request parameters or TLE data
- **422**: Validation error (missing required fields)
- **500**: Internal server error

Error response format:
```json
{
  "detail": "Error message describing the issue"
}
```

## Notes

- TLE lines must be at least 69 characters long
- Time window cannot exceed 168 hours (7 days)
- If satellite radii are not provided, random values between 1-10 meters are used
- The algorithm uses SGP4 propagation for accurate orbital predictions
- Initial filters may reject satellite pairs with very low collision risk

## Performance Considerations

- Typical response time: 100-500ms per prediction
- CPU-based computations (GPU support available if CUDA is installed)
- Consider implementing request rate limiting for production use
