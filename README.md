# AquaTrace AI
AI-powered marine oil spill detection, drift tracking, and potential vessel attribution using satellite imagery, oceanographic data, and AIS data.

### Marine Oil Spill Detection and Potential Vessel Attribution using Satellite Imagery and AIS Data

AquaTrace AI is an intelligent prototype designed to detect marine oil spills using satellite imagery, estimate the spill's movement and possible origin using oceanographic and meteorological data, and identify vessels that show strong spatial and temporal correlation with the estimated spill origin using historical AIS data.

## 🚨 Problem

Marine oil spills can cause severe damage to marine ecosystems, while identifying the vessel responsible can be difficult.

AquaTrace AI aims to combine:

- 🛰️ Satellite SAR imagery
- 🌊 Ocean current and wind data
- 🚢 Historical AIS vessel data
- 🤖 Machine learning and data-driven analysis

to create an automated oil-spill detection and vessel-attribution pipeline.

## 🔄 System Workflow

```text
Satellite SAR Image
        ↓
Oil Spill Detection
        ↓
Spill Characterization
        ↓
Ocean + Weather Data
        ↓
Forward / Backward Drift Simulation
        ↓
Estimated Spill Origin
        ↓
Historical AIS Data
        ↓
Vessel Filtering & Trajectory Analysis
        ↓
Potential Vessel Scoring
        ↓
Interactive Visualization
