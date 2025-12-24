# 🚀 Quick Start Guide - NearMiss UI

Get the NearMiss Satellite Collision Predictor UI up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install streamlit requests fastapi uvicorn
```

Or install everything:

```bash
pip install -r requirements.txt
```

## Step 2: Start the Application

### Option A: Start Everything at Once (Easiest)

```bash
cd src
python start_all.py
```

This starts both the API backend and the Streamlit UI automatically!

### Option B: Start Separately

**Terminal 1 - Start API Backend:**
```bash
cd src
python start_api_server.py
```

**Terminal 2 - Start Streamlit UI:**
```bash
cd src
python start_ui.py
```

## Step 3: Open Your Browser

The Streamlit UI will automatically open in your browser at:
```
http://localhost:8501
```

If it doesn't open automatically, navigate to that URL manually.

## Step 4: Make Your First Prediction

1. **Load Example Data**: Click the "📋 Load Example" buttons for both satellites

2. **Click Predict**: Hit the big red "🚀 PREDICT COLLISION RISK" button

3. **View Results**: See the collision probability, minimum distance, and risk assessment!

## That's It! 🎉

You now have a fully functional satellite collision prediction system with a beautiful UI!

## What You'll See

### Main Interface
- **Left Side**: Satellite 1 TLE inputs
- **Right Side**: Satellite 2 TLE inputs
- **Center**: Prediction button
- **Bottom**: Beautiful results with risk assessment

### Results Display
- 🔴 **High Risk**: >50% collision probability
- 🟡 **Moderate Risk**: 10-50% collision probability
- 🟢 **Low Risk**: <10% collision probability

### Sidebar Features
- API connection status (should show ✅ green)
- Device information (CPU/CUDA)
- Quick links to documentation

## Example TLE Data

If you need sample data, use these:

**Satellite 1:**
```
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
```

**Satellite 2:**
```
1 25545U 98067B   08264.52090000  .00000000  00000-0  00000-0 0  1234
2 25545  51.6416 247.5000 0006700 130.5000 325.0000 15.72125000563500
```

## Advanced Options

Click "⚙️ Advanced Options" to configure:
- **Time Window**: How many hours to analyze (1-168)
- **Satellite Radii**: Custom sizes in meters
- **Distance Threshold**: Minimum distance for collision consideration

## Common Issues

### "API is offline" in sidebar
**Fix**: Make sure the FastAPI backend is running
```bash
cd src
python start_api_server.py
```

### Port already in use
**Fix**: Use different ports
```bash
python start_ui.py --port 8502
python start_api_server.py --port 8001
```
(Then update API URL in sidebar to http://localhost:8001)

### Module not found
**Fix**: Install dependencies
```bash
pip install streamlit requests
```

## Next Steps

- 📖 Read the full [Streamlit UI Guide](STREAMLIT_UI_GUIDE.md)
- 🔧 Check out the [API Documentation](API_USAGE.md)
- 🧪 Try different time windows and satellite configurations
- 💾 Download your prediction results as JSON

## Quick Command Reference

| Command | Purpose |
|---------|---------|
| `python start_all.py` | Start everything at once |
| `python start_api_server.py` | Start only the API backend |
| `python start_ui.py` | Start only the Streamlit UI |
| `python start_ui.py --port 8502` | Start UI on custom port |

## Access Points

Once running, access these URLs:

- 🎨 **Streamlit UI**: http://localhost:8501
- 🔌 **API Backend**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

## Screenshots

### Main Interface
The UI features a clean, modern design with:
- Two-column layout for easy data entry
- Color-coded results
- Real-time API status
- One-click example data loading

### Results Display
Risk assessment with:
- Large, easy-to-read metrics
- Color-coded risk levels (🔴🟡🟢)
- Detailed analysis summary
- Downloadable JSON results

## Tips for Success

✅ **DO:**
- Use recent TLE data (less than 1 week old)
- Start with 24-hour time windows
- Check API status before predicting
- Save important high-risk predictions

❌ **DON'T:**
- Use TLE data older than 1 month
- Exceed 168-hour time windows
- Forget to start the API backend first

## Help & Support

Need help? Check these resources:

1. Full documentation: [docs/STREAMLIT_UI_GUIDE.md](STREAMLIT_UI_GUIDE.md)
2. API documentation: [docs/API_USAGE.md](API_USAGE.md)
3. FastAPI setup: [FASTAPI_SETUP.md](../FASTAPI_SETUP.md)

---

**Ready to predict satellite collisions? Let's go! 🛰️🚀**
