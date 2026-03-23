# 🚀 Streamlit UI - Quick Start Guide

## Overview

A user-friendly web interface for CTGAN synthetic data generation without requiring command-line usage.

## Features

✅ **Upload Data** - Drag-and-drop CSV upload or use sample data
✅ **Auto-Classification** - Automatic column type detection
✅ **Configure Model** - Easy parameter tuning with explanations
✅ **Train Model** - One-click training with progress tracking
✅ **View Results** - Interactive metrics with performance categorization
✅ **Download** - Export synthetic data and detailed reports

## Installation

### 1. Install Streamlit
```bash
pip install streamlit>=1.28.0
```

Or update requirements:
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
streamlit --version
# Should show: Streamlit, version 1.28.0 or higher
```

## Usage

### Launch the App

```bash
# Navigate to project directory
cd "c:\Users\c3058452\OneDrive - Newcastle University\Work in Progress\Aim 2"

# Run the Streamlit app
streamlit run streamlit_app.py
```

This will:
1. Start a local web server
2. Open your browser automatically to `http://localhost:8501`
3. Display the CTGAN UI

### Alternative: Run with Custom Configuration
```bash
# Run on different port
streamlit run streamlit_app.py --server.port 8502

# Run in developer mode
streamlit run streamlit_app.py --logger.level=debug
```

## Workflow

### Step 1: Upload Data 📤
1. **Upload CSV**: Click "Choose a CSV file" and select your data
   - Or use sample data buttons
2. **Preview**: See first 10 rows and statistics
3. **Check**: Monitor for missing values
4. **Classify**: Click "Analyze & Classify Columns"

### Step 2: Configure Model ⚙️
1. **Column Types**: Verify/adjust continuous, categorical, binary
2. **Condition Column**: (Optional) Select for conditional generation
3. **Hyperparameters**: 
   - Noise Dimension: 64-256
   - Batch Size: Auto-calculated
   - Learning Rates: G and D separately
4. **Architecture**: 
   - Generator hidden layers
   - Discriminator hidden layers
5. **Training**: 
   - Max Epochs: 10-500
   - Early Stopping: Patience 1-20
   - Min Delta: Convergence threshold
6. **Review**: See configuration summary

### Step 3: Train Model 🚀
1. **Start**: Click "Start Training" button
2. **Monitor**: Watch progress bar fill
3. **Wait**: Training usually takes 1-10 minutes
4. **Done**: See summary statistics

### Step 4: View Results 📊
1. **Metrics**: See RMSE, MSE, MAE, AUC
2. **Column Analysis**: 
   - [EXCELLENT] columns (expandable)
   - [GOOD] columns (expandable)
   - [FAIR] columns (expandable)
   - [POOR] columns (expandable, highlighted)
3. **Summary**: Success rate and column counts
4. **Sample**: View first 10 synthetic rows

### Step 5: Download 📥
1. **Synthetic Data**: Download as CSV
2. **Real Data**: Download cleaned version
3. **Report**: Download comprehensive report as TXT

## Interface Sections

### Sidebar Navigation
```
📋 Navigation
├─ 📤 Upload Data
├─ ⚙️ Configure Model
├─ 🚀 Train Model
├─ 📊 View Results
└─ 📥 Download
```

### Upload Data Page
- File uploader for CSV
- Sample data buttons
- Data preview (10 rows)
- Data statistics
- Missing value detection
- Auto-classification button

### Configure Model Page
- Column type selectors (multiselect)
- Condition column selector
- Hyperparameters (sliders)
- Model architecture (sliders)
- Training parameters (sliders)
- Configuration summary

### Train Model Page
- Configuration review
- Training progress bar
- Status updates
- Results summary

### View Results Page
- Metrics cards (RMSE, MSE, MAE, AUC)
- Column-wise analysis
- Performance categorization
- Sample synthetic data

### Download Page
- Synthetic data download
- Real data download
- Training report download

## Customization

### Colors & Styling
Edit the CSS section in `streamlit_app.py`:
```python
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;  # Change blue to your color
    }
    </style>
""", unsafe_allow_html=True)
```

### Default Parameters
Edit at top of Configure Model page:
```python
noise_dim = st.slider("Noise Dimension", value=64)  # Change default
epochs = st.slider("Max Epochs", value=100)  # Change default
```

### Column Names
Streamlit automatically picks from uploaded CSV - no configuration needed

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Install missing package
```bash
pip install streamlit
```

### Issue: Port 8501 already in use
**Solution**: Use different port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Slow upload for large files
**Solution**: Use smaller file or increase upload size limit
```bash
streamlit run streamlit_app.py --client.maxMessageSize 200
```

### Issue: Training crashes
**Solution**: Check error message, try with:
- Smaller batch size
- Fewer training parameters
- Different condition column
- Verify data has no extreme outliers

### Issue: "Session state" errors
**Solution**: Clear browser cache or use incognito mode

## Performance Tips

1. **Small Datasets** (< 1,000 rows)
   - Use epochs=50, patience=3
   - Model architecture: (128, 128)
   - Fast training: < 1 minute

2. **Medium Datasets** (1,000-100,000 rows)
   - Use epochs=100-200, patience=5
   - Model architecture: (256, 256)
   - Training: 2-10 minutes

3. **Large Datasets** (> 100,000 rows)
   - Use epochs=300+, patience=10
   - Model architecture: (512, 512)
   - Training: 30+ minutes
   - May need GPU for speed

## Advanced Usage

### Batch Processing
Save dataset configurations and run multiple trainings:
1. Upload and configure
2. Train model 1
3. Download results
4. Train model 2 with different config
5. Compare results

### Automated Workflows
Script with Streamlit CLI:
```bash
# Run with no browser
streamlit run streamlit_app.py --logger.level=off --client.showErrorDetails=false

# Run headless for automation
streamlit run streamlit_app.py --server.headless true
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB |
| CPU | 2 cores | 4+ cores |
| GPU | None | NVIDIA CUDA |

## Browser Compatibility

✅ Chrome (recommended)
✅ Firefox
✅ Safari
✅ Edge
⚠️ Mobile browsers (limited UI)

## File Structure

```
project/
├── streamlit_app.py          # Main Streamlit UI
├── CTGAN_dec_adjustable.py  # Core model
├── evaluation.py             # Metrics & analysis
├── data_loader.py            # Data handling
├── train_from_csv.py         # CLI training
├── requirements.txt          # Dependencies
└── sample_data.csv           # Sample data
```

## Security Notes

⚠️ **Local Use Only**: Default runs on localhost
⚠️ **Data Privacy**: All data processed locally, not uploaded
⚠️ **File Uploads**: Accepts CSV only
⚠️ **No Authentication**: Add if deploying to server

## Deployment (Advanced)

### Deploy to Streamlit Cloud
1. Push repo to GitHub
2. Go to https://share.streamlit.io
3. Deploy directly from repo
4. Share public link

### Deploy to Server
```bash
# Install production server
pip install gunicorn streamlit

# Run on server
gunicorn --bind 0.0.0.0:8000 streamlit_app.py
```

## Common Tasks

### Generate 50,000 rows from 5,000 real
1. Upload 5,000-row CSV
2. Configure (defaults OK)
3. Train model
4. Download synthetic data
5. Re-upload synthetic data as real
6. Retrain with different seed
7. Combine datasets

### Compare two conditions
1. Configure condition column (e.g., "Treatment")
2. Train model
3. View Results (see how each condition is handled)
4. Download and analyze

### Batch process multiple files
1. Create folder with CSVs
2. For each file:
   - Upload
   - Configure
   - Train
   - Download
   - Compare metrics

## Tips & Tricks

✨ **Tip 1**: Use sample data first to learn UI
✨ **Tip 2**: Start with default hyperparameters
✨ **Tip 3**: Check [POOR] columns first after training
✨ **Tip 4**: Success Rate > 75% is good target
✨ **Tip 5**: Use early stopping to save time
✨ **Tip 6**: Larger models help with complex data
✨ **Tip 7**: More epochs needed for imbalanced data

## Next Steps

1. **First Run**: Use sample data to explore UI
2. **Try Your Data**: Upload CSV and configure
3. **Train Model**: Start with default settings
4. **Review Results**: Check metrics and columns
5. **Download**: Export synthetic data
6. **Iterate**: Retrain with adjustments as needed

## Support

For issues:
1. Check Troubleshooting section above
2. Review console output for error messages
3. Try with sample data to isolate issue
4. Verify CSV format (columns should have names)
5. Check requirements installed correctly

## Documentation

Full documentation available in:
- `ENHANCED_REPORT_GUIDE.md` - Metrics explained
- `REPORT_QUICK_REFERENCE.md` - Quick lookup
- `AUTO_STOP_FEATURE.md` - Early stopping
- `TRAINING_REPORT_ENHANCEMENTS.md` - Report details

## Summary

✨ **Streamlit UI makes CTGAN accessible to non-technical users:**
- No command-line knowledge required
- Interactive parameter tuning
- Real-time feedback and progress
- Professional quality output
- Easy data export

**Ready to generate high-quality synthetic data!** 🚀
