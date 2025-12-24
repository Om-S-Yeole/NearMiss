"""
FastAPI application for satellite collision prediction.

This module provides a REST API endpoint for predicting collision probability
and minimum distance between two satellites using their TLE data.
"""

from datetime import datetime, timedelta
from typing import Optional

import pytz
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from nearmiss.astro.close_approach_algorithm_sgp4 import (
    close_approach_physical_algorithm_sgp4,
)

# Initialize FastAPI app
app = FastAPI(
    title="NearMiss Satellite Collision Prediction API",
    description="API for predicting satellite collision risks using TLE data",
    version="1.0.0",
)


# Request and Response Models
class SatelliteTLE(BaseModel):
    """Model for satellite TLE data."""

    tle_line1: str = Field(..., description="TLE line 1 for the satellite")
    tle_line2: str = Field(..., description="TLE line 2 for the satellite")

    @field_validator("tle_line1", "tle_line2")
    @classmethod
    def validate_tle_line(cls, v: str) -> str:
        """Validate TLE line format."""
        if not v or len(v.strip()) < 69:
            raise ValueError("TLE line must be at least 69 characters long")
        return v.strip()


class PredictionRequest(BaseModel):
    """Model for collision prediction request."""

    satellite1: SatelliteTLE = Field(..., description="TLE data for satellite 1")
    satellite2: SatelliteTLE = Field(..., description="TLE data for satellite 2")
    time_window_hours: Optional[float] = Field(
        default=24.0,
        description="Time window for analysis in hours (max 168 hours = 7 days)",
        gt=0,
        le=168,
    )
    satellite1_radius: Optional[float] = Field(
        default=None, description="Radius of satellite 1 in meters", gt=0
    )
    satellite2_radius: Optional[float] = Field(
        default=None, description="Radius of satellite 2 in meters", gt=0
    )
    distance_threshold: Optional[float] = Field(
        default=10.0, description="Minimum distance threshold in kilometers", ge=0
    )


class PredictionResponse(BaseModel):
    """Model for collision prediction response."""

    collision_probability: float = Field(
        ..., description="Probability of collision (0 to 1)"
    )
    minimum_distance_km: float = Field(
        ..., description="Minimum distance between satellites in kilometers"
    )
    filter_rejected: bool = Field(
        ..., description="Whether the pair was rejected by initial filters"
    )
    analysis_start_time: str = Field(..., description="Start time of analysis window")
    analysis_end_time: str = Field(..., description="End time of analysis window")
    message: str = Field(..., description="Additional information about the prediction")


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "NearMiss Satellite Collision Prediction API",
        "version": "1.0.0",
        "description": "Predict collision probability between satellites using TLE data",
        "endpoints": {
            "/predict": "POST - Predict collision between two satellites",
            "/health": "GET - Check API health status",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_collision(request: PredictionRequest):
    """
    Predict collision probability and minimum distance between two satellites.

    This endpoint uses SGP4 propagation and physical algorithms to calculate
    the closest approach and collision probability between two satellites
    within a specified time window.

    Parameters
    ----------
    request : PredictionRequest
        Request containing TLE data for both satellites and analysis parameters.

    Returns
    -------
    PredictionResponse
        Response containing collision probability, minimum distance, and analysis details.

    Raises
    ------
    HTTPException
        400: Invalid TLE data or parameters
        500: Internal server error during computation
    """
    try:
        # Parse TLE data
        tle1 = (request.satellite1.tle_line1, request.satellite1.tle_line2)
        tle2 = (request.satellite2.tle_line1, request.satellite2.tle_line2)

        # Set analysis time window
        D_start = datetime.now().replace(tzinfo=pytz.utc)
        D_stop = D_start + timedelta(hours=request.time_window_hours)

        # Determine satellite radii
        random_sat_radii = (
            request.satellite1_radius is None or request.satellite2_radius is None
        )
        r_obj_1 = request.satellite1_radius
        r_obj_2 = request.satellite2_radius

        # Run collision analysis algorithm
        result = close_approach_physical_algorithm_sgp4(
            tle1=tle1,
            tle2=tle2,
            D_start=D_start,
            D_stop=D_stop,
            random_sat_radii=random_sat_radii,
            r_obj_1=r_obj_1,
            r_obj_2=r_obj_2,
            Dist=request.distance_threshold,
        )

        # Extract results
        filter_rejected = result.output.filter_rej_code != 0
        ln_d_min = result.output.ln_d_min
        collision_probability = result.output.probab

        # Calculate actual minimum distance from ln(1 + d_min)
        import math

        minimum_distance_km = math.exp(ln_d_min) - 1 if ln_d_min > 0 else 0.0

        # Prepare response message
        if filter_rejected:
            message = "Satellite pair rejected by initial filters. Low collision risk."
        elif collision_probability > 0.5:
            message = f"High collision risk detected! Minimum distance: {minimum_distance_km:.2f} km"
        elif collision_probability > 0.1:
            message = f"Moderate collision risk. Minimum distance: {minimum_distance_km:.2f} km"
        else:
            message = (
                f"Low collision risk. Minimum distance: {minimum_distance_km:.2f} km"
            )

        return PredictionResponse(
            collision_probability=collision_probability,
            minimum_distance_km=minimum_distance_km,
            filter_rejected=filter_rejected,
            analysis_start_time=D_start.isoformat(),
            analysis_end_time=D_stop.isoformat(),
            message=message,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid input parameters: {str(e)}"
        )
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data type: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
