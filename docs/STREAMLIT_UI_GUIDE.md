# Streamlit UI Guide for NearMiss

## Overview

The NearMiss Streamlit UI provides a beautiful, user-friendly interface for predicting satellite collision risks. It connects to the FastAPI backend to perform predictions and displays results in an intuitive, visually appealing format.

## Features

✨ **Beautiful UI Design**
- Modern gradient color schemes
- Responsive layout
- Interactive components
- Real-time API status monitoring

🎯 **Core Functionality**
- Input TLE data for two satellites
- Configure analysis parameters (time window, radii, distance threshold)
- One-click prediction with visual feedback
- Color-coded risk assessment (Green/Yellow/Red)
- Detailed results with metrics cards

📊 **Results Display**
- Collision probability percentage
- Minimum distance in kilometers
- Filter status (passed/rejected)
- Analysis time window
- Risk interpretation and recommendations
- JSON export capability

## Installation

### Prerequisites

Make sure you have Python 3.8+ installed.

### Install Dependencies

```bash
pip install streamlit requests
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

## Running the UI

### Option 1: Using the Startup Script (Recommended)

Start the UI only:

```bash
cd src
python start_ui.py
```

With custom port:

```bash
python start_ui.py --port 8502
```

### Option 2: Start Both Backend and UI

Start both FastAPI backend and Streamlit UI simultaneously:

```bash
cd src
python start_all.py
```

This will start:
- FastAPI on http://localhost:8000
- Streamlit UI on http://localhost:8501

### Option 3: Direct Streamlit Command

```bash
cd src
streamlit run ui/streamlit_app.py
```

## Using the Application

### 1. Launch the Application

After starting the UI, your browser should automatically open to `http://localhost:8501`

If it doesn't, manually navigate to that URL.

### 2. Check API Status

The sidebar shows the API connection status:
- ✅ **Green**: API is online and ready
- ❌ **Red**: API is offline (start the backend first)
- ⚠️ **Yellow**: API connection issue

### 3. Enter Satellite Data

**Satellite 1 (Left Column):**
- Enter TLE Line 1 (69+ characters)
- Enter TLE Line 2 (69+ characters)
- Or click "📋 Load Example 1" to use sample data

**Satellite 2 (Right Column):**
- Enter TLE Line 1 (69+ characters)
- Enter TLE Line 2 (69+ characters)
- Or click "📋 Load Example 2" to use sample data

**Example TLE Data:**

Satellite 1 (ISS):
```
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
```

Satellite 2:
```
1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234
2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500
```

### 4. Configure Advanced Options (Optional)

Click "⚙️ Advanced Options" to expand:

- **Time Window**: 1-168 hours (default: 24 hours)
- **Custom Satellite Radii**: Specify exact sizes in meters
  - Enable checkbox to input custom values
  - Satellite 1 Radius: 0.1-100 meters
  - Satellite 2 Radius: 0.1-100 meters
- **Distance Threshold**: 1-100 km (default: 10 km)

### 5. Run Prediction

Click the **"🚀 PREDICT COLLISION RISK"** button.

The system will:
1. Validate all inputs
2. Send request to FastAPI backend
3. Display loading spinner during calculation
4. Show results when complete

### 6. Interpret Results

The results are displayed in three main metric cards:

**🔴 HIGH RISK (>50% probability)**
- Immediate attention required
- Detailed collision avoidance analysis needed
- Consider orbital maneuvers

**🟡 MODERATE RISK (10-50% probability)**
- Monitor closely
- Prepare contingency plans
- Increase observation frequency

**🟢 LOW RISK (<10% probability)**
- Normal monitoring sufficient
- Standard operational procedures

**Additional Information:**
- **Minimum Distance**: Closest approach in kilometers
- **Filter Status**: Initial orbital screening result
- **Analysis Summary**: Human-readable risk assessment
- **Time Window**: Analysis period details

### 7. Download Results

Click **"📥 Download Results (JSON)"** to save the prediction data in JSON format.

The file includes:
- Collision probability
- Minimum distance
- Filter status
- Analysis timestamps
- Risk message

## UI Components

### Sidebar

**Configuration Section:**
- API URL input (default: http://localhost:8000)
- Real-time API health status
- Device information (CPU/CUDA)

**About Section:**
- Technology stack information
- Quick links to API documentation

### Main Area

**Input Section:**
- Two-column layout for satellite TLE data
- Example data loading buttons
- Input validation with helpful tooltips

**Advanced Options:**
- Collapsible expander
- Time window slider
- Custom radii toggle and inputs
- Distance threshold configuration

**Results Section:**
- Three metric cards with gradient backgrounds
- Color-coded risk levels
- Analysis summary box
- Expandable details section
- Download button

## Troubleshooting

### API Connection Error

**Problem:** "❌ API is offline" or "Cannot connect to API"

**Solution:**
1. Make sure the FastAPI backend is running:
   ```bash
   cd src
   python start_api_server.py
   ```
2. Check if the API URL in the sidebar is correct
3. Verify port 8000 is not blocked by firewall

### Invalid TLE Data

**Problem:** "TLE lines must be at least 69 characters long"

**Solution:**
- Ensure TLE lines are complete
- Remove any extra spaces at the beginning
- Use the example data buttons to test

### Timeout Error

**Problem:** "Request timed out"

**Solution:**
- The calculation may take time for complex orbits
- Try reducing the time window
- Check if the API server is overloaded

### Port Already in Use

**Problem:** "Address already in use"

**Solution:**
```bash
# Use a different port
python start_ui.py --port 8502
```

### Module Not Found

**Problem:** "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit requests
```

## Keyboard Shortcuts

- **Ctrl+C**: Stop the UI server (in terminal)
- **Ctrl+R**: Refresh the UI (in browser)
- **Ctrl+S**: Save (when editing text fields)

## Tips for Best Results

1. **Use Recent TLE Data**: TLE data becomes less accurate over time. Use data that's less than a week old.

2. **Appropriate Time Windows**: 
   - Short-term predictions: 6-24 hours
   - Medium-term: 24-72 hours
   - Long-term: up to 7 days (less accurate)

3. **Satellite Radii**: If you know the actual sizes, use custom radii for more accurate results.

4. **Multiple Predictions**: Run predictions with different time windows to see how risk evolves.

5. **Save Important Results**: Use the download button to keep records of high-risk predictions.

## Advanced Usage

### Running on a Different Host

To make the UI accessible from other machines on your network:

```bash
python start_ui.py --host 0.0.0.0 --port 8501
```

Then access it at: `http://YOUR_IP_ADDRESS:8501`

### Custom API URL

If your FastAPI backend is running on a different machine or port, update the API URL in the sidebar.

Example: `http://192.168.1.100:8000`

## Technical Details

**Technologies Used:**
- **Streamlit**: Web UI framework
- **Requests**: HTTP client for API calls
- **Custom CSS**: Enhanced styling with gradients and animations

**UI Components:**
- Text inputs with validation
- Sliders and number inputs
- Buttons with hover effects
- Expanders for advanced options
- Metric cards with gradients
- Download functionality

**API Integration:**
- Health check on startup
- POST requests to /predict endpoint
- Error handling and user feedback
- Timeout management (30 seconds)

## Customization

To customize the UI appearance, edit the CSS in [src/ui/streamlit_app.py](../src/ui/streamlit_app.py):

```python
st.markdown("""
    <style>
    /* Add your custom CSS here */
    </style>
""", unsafe_allow_html=True)
```

## Support

For issues or questions:
1. Check the API is running: http://localhost:8000/health
2. Review the API documentation: http://localhost:8000/docs
3. Check console logs for error messages
4. Verify all dependencies are installed

## Next Steps

- Try predicting collisions with real satellite data
- Experiment with different time windows
- Export and analyze historical predictions
- Integrate with your orbital tracking system

Enjoy using the NearMiss Satellite Collision Predictor! 🛰️
