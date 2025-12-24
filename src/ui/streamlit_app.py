"""
Streamlit UI for NearMiss Satellite Collision Prediction.

This application provides a user-friendly interface for predicting satellite
collision risks using TLE data through the FastAPI backend.
"""

import requests
import streamlit as st
from datetime import datetime
import json


# Page configuration
st.set_page_config(
    page_title="NearMiss - Satellite Collision Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.02);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .info-box h3 {
        color: #667eea;
        margin-top: 0;
    }
    .info-box p {
        color: #333333;
    }
    h1 {
        color: #667eea;
        font-weight: 700;
    }
    h2 {
        color: #764ba2;
    }
    h3 {
        color: #667eea;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def check_api_health(api_url: str) -> dict:
    """Check if the FastAPI backend is running and healthy."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "unhealthy", "error": "API returned non-200 status"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "error": "Cannot connect to API"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def predict_collision(api_url: str, payload: dict) -> dict:
    """Send prediction request to the FastAPI backend."""
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=30)

        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return {"status": "error", "error": error_detail}
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Request timed out. The server may be busy.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "error": "Cannot connect to API. Make sure the server is running.",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def display_results(result: dict):
    """Display prediction results in a beautiful format."""
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    collision_prob = result["collision_probability"]
    min_distance = result["minimum_distance_km"]
    filter_rejected = result["filter_rejected"]
    message = result["message"]

    # Display main metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # Collision probability with color coding
        if collision_prob > 0.5:
            color = "🔴"
            risk_level = "HIGH RISK"
        elif collision_prob > 0.1:
            color = "🟡"
            risk_level = "MODERATE RISK"
        else:
            color = "🟢"
            risk_level = "LOW RISK"

        st.markdown(
            f"""
        <div class="metric-card">
            <h2>{color} {risk_level}</h2>
            <h1>{collision_prob:.4%}</h1>
            <p>Collision Probability</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="success-card">
            <h2>🛰️ Minimum Distance</h2>
            <h1>{min_distance:.2f} km</h1>
            <p>Closest Approach</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        filter_status = "✅ PASSED" if not filter_rejected else "❌ REJECTED"
        filter_color = "success-card" if not filter_rejected else "warning-card"
        st.markdown(
            f"""
        <div class="{filter_color}">
            <h2>🔍 Filter Status</h2>
            <h1>{filter_status}</h1>
            <p>Initial Screening</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Display message
    st.markdown(
        f"""
    <div class="info-box">
        <h3>📝 Analysis Summary</h3>
        <p style="font-size: 16px; margin: 0;">{message}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Additional details in expander
    with st.expander("📅 View Analysis Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Analysis Start:**")
            st.text(result["analysis_start_time"])
        with col2:
            st.markdown("**Analysis End:**")
            st.text(result["analysis_end_time"])

        # Risk interpretation
        st.markdown("---")
        st.markdown("**Risk Interpretation:**")
        if filter_rejected:
            st.info(
                "The satellite pair was rejected by initial orbital filters, indicating very low collision risk due to non-intersecting orbital paths."
            )
        elif collision_prob > 0.5:
            st.error(
                f"⚠️ HIGH COLLISION RISK! The satellites may pass within {min_distance:.2f} km of each other. Immediate attention and detailed analysis recommended."
            )
        elif collision_prob > 0.1:
            st.warning(
                f"Moderate collision risk detected. Monitor these satellites closely. Minimum approach distance: {min_distance:.2f} km"
            )
        else:
            st.success(
                f"Low collision risk. Satellites will maintain safe distance of at least {min_distance:.2f} km"
            )


def main():
    """Main Streamlit application."""

    # Header
    st.markdown(
        """
        <h1 style='text-align: center;'>
            🛰️ NearMiss Satellite Collision Predictor
        </h1>
        <p style='text-align: center; font-size: 18px; color: #666;'>
            Predict collision risks between satellites using TLE data and SGP4 propagation
        </p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Sidebar for API configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        api_url = st.text_input(
            "API URL", value="http://localhost:8000", help="URL of the FastAPI backend"
        )

        # Check API health
        health = check_api_health(api_url)
        if health["status"] == "healthy":
            st.success("✅ API is online and healthy")
            if "data" in health:
                device = health["data"].get("device", "unknown")
                st.info(f"🖥️ Running on: {device}")
        elif health["status"] == "offline":
            st.error("❌ API is offline")
            st.warning(
                "Please start the FastAPI server:\n```\ncd src\npython start_api_server.py\n```"
            )
        else:
            st.warning(f"⚠️ API Status: {health['status']}")

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown(
            """
        This tool uses:
        - **SGP4** propagation
        - Physical collision algorithms
        - Neural network models
        
        for accurate satellite collision risk assessment.
        """
        )

        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("[📚 API Docs](http://localhost:8000/docs)")
        st.markdown("[❤️ Health Check](http://localhost:8000/health)")

    # Main content area
    st.markdown("## 🎯 Enter Satellite Data")

    # TLE Input Section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛰️ Satellite 1")
        sat1_line1 = st.text_input(
            "TLE Line 1",
            key="sat1_line1",
            placeholder="1 44449U 19045A   25357.31513255  .00004050  00000+0  30815-3 0  9999",
            help="First line of Two-Line Element set (69+ characters)",
        )
        sat1_line2 = st.text_input(
            "TLE Line 2",
            key="sat1_line2",
            placeholder="2 44449  34.9957 321.2135 0004876 202.8341 157.2150 15.01588635350154",
            help="Second line of Two-Line Element set (69+ characters)",
        )

    with col2:
        st.markdown("### 🛰️ Satellite 2")
        sat2_line1 = st.text_input(
            "TLE Line 1",
            key="sat2_line1",
            placeholder="1 58748U 24005W   25357.17682812  .00000142  00000+0  25602-4 0  9995",
            help="First line of Two-Line Element set (69+ characters)",
        )
        sat2_line2 = st.text_input(
            "TLE Line 2",
            key="sat2_line2",
            placeholder="2 58748  43.0042 277.2461 0001656 262.5004  97.5651 15.02536392108914",
            help="Second line of Two-Line Element set (69+ characters)",
        )

    st.markdown("---")

    # Advanced Options
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            time_window = st.slider(
                "Time Window (hours)",
                min_value=1.0,
                max_value=168.0,
                value=24.0,
                step=1.0,
                help="Analysis time window (max 168 hours = 7 days)",
            )

        with col2:
            use_custom_radii = st.checkbox(
                "Use Custom Satellite Radii",
                value=False,
                help="Specify exact radii instead of random values",
            )

            if use_custom_radii:
                sat1_radius = st.number_input(
                    "Satellite 1 Radius (m)",
                    min_value=0.1,
                    max_value=100.0,
                    value=5.0,
                    step=0.1,
                )
                sat2_radius = st.number_input(
                    "Satellite 2 Radius (m)",
                    min_value=0.1,
                    max_value=100.0,
                    value=5.0,
                    step=0.1,
                )
            else:
                sat1_radius = None
                sat2_radius = None

        with col3:
            distance_threshold = st.number_input(
                "Distance Threshold (km)",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                help="Minimum distance threshold for collision consideration",
            )

    st.markdown("---")

    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🚀 PREDICT COLLISION RISK", use_container_width=True
        )

    # Handle prediction
    if predict_button:
        # Validate inputs
        if not all([sat1_line1, sat1_line2, sat2_line1, sat2_line2]):
            st.error("❌ Please fill in all TLE fields for both satellites!")
            return

        if len(sat1_line1.strip()) < 69 or len(sat1_line2.strip()) < 69:
            st.error("❌ Satellite 1 TLE lines must be at least 69 characters long!")
            return

        if len(sat2_line1.strip()) < 69 or len(sat2_line2.strip()) < 69:
            st.error("❌ Satellite 2 TLE lines must be at least 69 characters long!")
            return

        # Prepare payload
        payload = {
            "satellite1": {
                "tle_line1": sat1_line1.strip(),
                "tle_line2": sat1_line2.strip(),
            },
            "satellite2": {
                "tle_line1": sat2_line1.strip(),
                "tle_line2": sat2_line2.strip(),
            },
            "time_window_hours": time_window,
            "distance_threshold": distance_threshold,
        }

        if use_custom_radii and sat1_radius and sat2_radius:
            payload["satellite1_radius"] = sat1_radius
            payload["satellite2_radius"] = sat2_radius

        # Show loading spinner
        with st.spinner(
            "🔄 Analyzing orbital trajectories and calculating collision risk..."
        ):
            result = predict_collision(api_url, payload)

        # Display results
        if result["status"] == "success":
            display_results(result["data"])

            # Option to download results
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                json_str = json.dumps(result["data"], indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_str,
                    file_name=f"collision_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.error(f"❌ Error: {result['error']}")
            if "Cannot connect to API" in result["error"]:
                st.info(
                    "💡 Make sure the FastAPI server is running. Start it with:\n```\ncd src\npython start_api_server.py\n```"
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🛰️ NearMiss Satellite Collision Predictor v1.0.0</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
